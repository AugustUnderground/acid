{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE RecordWildCards #-}

-- | Artificial Circuit Designer
module ACiD where

import           Control.Monad
import           Lib
import           CKT
import           MLFlow
import           MLFlow.Extensions
import           ALG
import           ALG.HyperParameters
import           RPB
-- import qualified Data.Set            as S
import qualified ALG.TD3             as TD3
import qualified RPB.HER             as HER
import qualified Torch               as T
import qualified Torch.Extensions    as T

-- | Policy evaluation Episode
eval :: (Agent a) => CircusUrl -> a -> Int -> T.Tensor -> Float -> Bool 
     -> IO Float
eval _    _     _ _   success True  = pure success
eval addr agent t obs success False = do
    (!state',_,!goal,!reward,!done) <- step addr $ act'' agent obs

    let success' = successRate done reward
        done'    = T.all done
        obs'     = T.cat (T.Dim 1) [state', goal]
    
    when (verbose && t `mod` 10 == 0) do
        putStrLn $ "\tStep " ++ show t ++ ":"
        putStrLn $ "\t\tSuccess Rate: " ++ show success ++ "%"

    eval addr agent t' obs' success' done'
  where
    t' = t + 1

-- | Runs training on given Agent with Buffer
train :: (Agent a, ReplayBuffer b) => CircusUrl -> Tracker -> String -> Int 
      -> b T.Tensor -> a -> IO ()
train _    _       path 0       _      agent = void $ saveAgent path agent
train addr tracker path episode buffer agent = do

    when verbose do
        let isRandom = if iter `mod` explFreq == 0 
                          then "Random Exploration" 
                          else "Policy Exploitation"
        putStrLn $ "Episode " ++ show iter ++ " (" ++ isRandom ++ "):"

    buffer'  <-  push bufferSize buffer 
             <$> collectExperience addr tracker iter agent

    batches  <-  map (tmap (T.toDevice T.gpu)) 
             <$> randomBatches numEpochs batchSize buffer'

    agent'   <-  updatePolicy addr tracker iter batches agent >>= saveAgent path

    when (iter /= 0 && iter `mod` evalFreq == 0) do
        (state,_,goal) <- reset addr
        let obs   = T.cat (T.Dim 1) [state, goal]
            iter' = iter `div` evalFreq
        putStrLn $ "Policy Evaluation Episode " ++ show iter'
        success   <- eval addr agent 0 obs 0.0 False
        _         <- trackLoss tracker iter' "Success" success
        pure ()

    train addr tracker path episode' buffer' agent'
  where
    episode' =   episode - 1
    iter     =   numEpisodes - episode

-- | Create Agent and Buffer, then run training
run' :: CircusUrl -> Tracker -> String -> Mode -> Algorithm -> ReplayMemory 
    -> IO ()
run' addr tracker path Train TD3 HER = do
    actDim     <- actionSpace addr
    (od,gd,_)  <- observationSpace addr
    let obsDim =  od + gd

    agent       <- TD3.mkAgent obsDim actDim
    let buffer  =  HER.empty

    train addr tracker path numEpisodes buffer agent
run' _    _       _    _     _   _   = error "Not Implemented"

-- | Clown School
run :: Args -> IO ()
run Args{..} = do
    nEnvs   <- numEnvs url'
    tracker <- mkTracker uri' expName >>= newRuns' nEnvs
    path'   <- createModelArchiveDir' cpPath algorithm ace pdk var space
    run' url' tracker path' mode' alg buf
  where
    alg     = read algorithm :: Algorithm
    buf     = read memory    :: ReplayMemory
    mode'   = read mode      :: Mode
    -- space'  = read space     :: Space
    url'    = url cktHost cktPort ace pdk space var
    uri'    = trackingURI mlfHost mlfPort
    expName = algorithm ++ "-" ++ memory ++ "-" ++ ace ++ "-" ++ pdk
                        ++ "-" ++ space ++ "-v" ++ var ++ "-" ++ mode
