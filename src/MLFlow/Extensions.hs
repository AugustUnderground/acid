{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE BlockArguments  #-}
{-# LANGUAGE RecordWildCards #-}

-- | ACiD Specific Extensions to the mlflow-hs package
module MLFlow.Extensions where

import qualified Torch                 as T
import qualified Data.Map              as M
import           Data.Maybe                    (fromJust)
import           Data.Time.Clock.POSIX         (getPOSIXTime)
import           Control.Monad
import           Network.Wreq          as Wreq
import qualified Data.ByteString.Lazy  as BL
import qualified MLFlow                as MLF
import qualified MLFlow.DataStructures as MLF

import qualified CKT

------------------------------------------------------------------------------
-- Data Logging / Visualization
------------------------------------------------------------------------------

-- | Sanatize JSON for MLFlow: Names may only contain alphanumerics,
-- underscores (_), dashes (-), periods (.), spaces ( ), and slashes (/).
sanatizeJSON :: Char -> Char
sanatizeJSON ':' = '_'  -- Replace Colons
sanatizeJSON ';' = '_'  -- Replace Semicolons
sanatizeJSON ',' = '_'  -- Replace Commas
sanatizeJSON  c  =  c   -- Leave as is

-- | Data Logging to MLFlow Trackign Server
data Tracker = Tracker { uri            :: MLF.TrackingURI        -- ^ Tracking Server URI
                       , experimentId   :: MLF.ExperimentID       -- ^ Experiment ID
                       , experimentName :: String                 -- ^ Experiment Name
                       , runIds         :: M.Map String MLF.RunID -- ^ Run IDs
                       } deriving (Show)

-- | Retrieve a run ID
runId :: Tracker -> String -> MLF.RunID
runId Tracker{..} id' = fromJust $ M.lookup id' runIds

-- | Make new Tracker given a Tracking Server URI
mkTracker :: MLF.TrackingURI -> String -> IO Tracker
mkTracker uri' expName = do
    suffix <- (round . (* 1000) <$> getPOSIXTime :: IO Int)
    let expName' = expName ++ "_" ++ show suffix
    expId' <- MLF.createExperiment uri' expName'
    pure (Tracker uri' expId' expName' M.empty)

-- | Make new Tracker given a Hostname and Port
mkTracker' :: String -> Int -> String -> IO Tracker
mkTracker' host port = mkTracker (MLF.trackingURI' host port)

-- | Create a new Experiment with rng suffix
newExperiment :: Tracker -> String -> IO Tracker
newExperiment Tracker{..} expName = do
    suffix <- (round . (* 1000) <$> getPOSIXTime :: IO Int)
    let expName' = expName ++ "_" ++ show suffix
    expId' <- MLF.createExperiment uri expName'
    pure (Tracker uri expId' expName' M.empty)

-- | Create a new Experiment
newExperiment' :: Tracker -> String -> IO Tracker
newExperiment' Tracker{..} expName = do
    expId' <- MLF.createExperiment uri expName
    pure (Tracker uri expId' expName M.empty)

-- | Create a new run with a set of given paramters
newRuns :: Tracker -> [String] -> [MLF.Param] -> IO Tracker
newRuns Tracker{..} ids params' = do
    unless (M.null runIds) do
        forM_ (M.elems runIds) (MLF.endRun uri)
        putStrLn "Ended runs before starting new ones."
    runIds' <- replicateM (length ids) 
                 (MLF.runId . MLF.runInfo <$> MLF.createRun uri experimentId [])
    forM_ (zip runIds' params') (\(rid, p') -> MLF.logBatch uri rid [] [p'] [])
    let runs = M.fromList $ zip ids runIds'
    pure (Tracker uri experimentId experimentName runs)

-- | New run with algorithm id and #envs as log params
newRuns' :: Int -> Tracker -> IO Tracker
newRuns' numEnvs tracker = newRuns tracker ids params'
  where
    ids     = "model" 
            : map (("env_" ++) . show) [0 .. (numEnvs - 1)] 
    params' = MLF.Param "id" "model" 
            : map (MLF.Param "id" . show) [0 .. (numEnvs - 1)] 

-- | End a run
endRun :: String -> Tracker -> IO Tracker
endRun id' tracker@Tracker{..} = do
    _ <- MLF.endRun uri (runId tracker id')
    pure (Tracker uri experimentId experimentName runIds')
  where 
    runIds' = M.delete id' runIds

-- | End all runs of a Tracker
endRuns :: Tracker -> IO Tracker
endRuns tracker@Tracker{..} = do
    let _ = M.map (MLF.endRun uri . runId tracker) runIds
    pure (Tracker uri experimentId experimentName M.empty)

-- | End all runs and discard tracker
endRuns' :: Tracker -> IO ()
endRuns' tracker = do
    _ <- endRuns tracker
    pure ()

-- | Write Loss to Tracking Server
trackLoss :: Tracker -> Int -> String -> Float -> IO (Response BL.ByteString)
trackLoss tracker@Tracker{..} epoch ident loss = 
    MLF.logMetric uri runId' ident loss epoch
  where
    runId' = runId tracker "model" 

-- | Write Reward to Tracking Server
trackReward :: Tracker -> Int -> T.Tensor -> IO ()
trackReward tracker@Tracker{..} step reward = do
        _ <- MLF.logMetric uri rewId "sum" rSum step
        _ <- MLF.logMetric uri rewId "avg" rAvg step
        _ <- MLF.logMetric uri rewId "max" rMax step
        _ <- MLF.logMetric uri rewId "min" rMin step
        forM_ (zip envIds rewards) 
            (\(envId, rewardValue) -> 
                let runId' = runId tracker envId
                 in MLF.logMetric uri runId' "reward" rewardValue step)
  where
    rewards = T.asValue (T.squeezeAll reward) :: [Float]
    envIds  = [ "env_" ++ show e | e <- [0 .. (length rewards - 1) ]]
    rewId   = runId tracker "model"
    rAvg    = T.asValue (T.mean reward) :: Float
    rSum    = T.asValue (T.sumAll reward) :: Float
    rMin    = T.asValue (T.min reward) :: Float
    rMax    = T.asValue (T.max reward) :: Float
                    
-- | Clean up a Map returned by Circus Server
sanatizeMap :: M.Map Int (M.Map String Float) -> M.Map Int (M.Map String Float)
sanatizeMap = M.map (M.mapKeys $ map sanatizeJSON)

-- | Write Current state of the Environment to Trackign Server
trackEnvState :: Tracker -> CKT.CircusUrl -> Int -> IO ()
trackEnvState tracker@Tracker{..} url step = do

    performance <- sanatizeMap <$> CKT.currentPerformance url
    sizing      <- sanatizeMap <$> CKT.currentSizing url
    actions     <- sanatizeMap <$> CKT.lastAction url 
    goal        <- M.map (M.mapKeys (++ "_target")) . sanatizeMap 
               <$> CKT.currentGoal url

    forM_ (M.keys goal)
          (\id' -> 
              let envId  = "env_" ++ show id'
                  state  = M.unions 
                         $ map (fromJust . M.lookup id')
                               [goal, performance, sizing, actions]
                  runId' = runId tracker envId
               in MLF.logBatch' uri runId' step state M.empty)
