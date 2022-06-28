{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE BangPatterns    #-}
{-# LANGUAGE BlockArguments  #-}
{-# LANGUAGE RecordWildCards #-}

-- | Artificial Circuit Designer
module ACiD where

import           Control.Monad
import           Lib
import           CKT
import           HyperParameters
import           MLFlow
import           MLFlow.Extensions
import           ALG
import           RPB
import qualified ALG.TD3             as TD3
import qualified RPB.HER             as HER
import qualified Torch               as T
import qualified Torch.Extensions    as T

-- | Policy evaluation Episode
eval :: (Agent a) => Params -> CircusUrl -> Tracker -> a 
     -> Int -> Int -> T.Tensor -> T.Tensor -> IO ()
eval p@Params{..} addr tracker agent episode t obs dones | T.all dones = pure ()
                                                         | otherwise   = do
    (!state',_,!goal,!reward,!done) <- step addr $ act'' agent obs

    let dones'  = T.logicalOr done dones
        obs'    = T.cat (T.Dim 1) [state', goal]
        success = successRate . T.logicalOr dones . T.logicalAnd done 
                $ T.ge reward 0.0
    
    when (t % 10 == 0) do
        putStrLn $ "\tStep " ++ show t ++ ":"
        putStrLn $ "\t\tSuccess Rate: " ++ show success ++ "%"

    _ <- trackLoss tracker (iter' !! t) "Success" success

    trackEnvState tracker addr t'
    eval p addr tracker agent episode t' obs' dones'
  where
    t'    = t + 1
    iter' = map ((episode * horizonT) +) [0..]

-- | Runs training on given Agent with Buffer
train :: (Agent a, ReplayBuffer b) => Params -> CircusUrl -> Tracker -> String 
      -> Int -> b T.Tensor -> a -> IO ()
train _            _    _       path 0       _   agent = void $ saveAgent path agent
train p@Params{..} addr tracker path episode buf agent = do

    let isRandom = if iter % explFreq == 0 
                      then "Random Exploration" 
                      else "Policy Exploitation"
    putStrLn $ "Episode " ++ show iter ++ " / " ++ show numEpisodes 
                          ++ " (" ++ isRandom ++ "):"

    buf'  <-  push bufferSize buf 
             <$> collectExperience p addr tracker iter agent

    batches  <-  map (tmap (T.toDevice T.gpu)) 
             <$> randomBatches numEpochs batchSize buf'

    agent'   <-  updatePolicy p addr tracker iter batches agent 
                    >>= saveAgent path

    --when (iter /= 0 && iter % evalFreq == 0) do
    --    (state,_,goal) <- reset addr
    --    let obs      = T.cat (T.Dim 1) [state, goal]
    --        iter'    = iter // evalFreq
    --        success' = T.full [head $ T.shape obs] False 
    --                 $ T.withDType T.Bool T.defaultOpts
    --    putStrLn $ "Policy Evaluation Episode " ++ show iter'
    --    eval p addr tracker agent' iter' 0 obs success'
    --    pure ()

    train p addr tracker path episode' buf' agent'
  where
    episode' = episode - 1
    iter     = numEpisodes - episode

-- | Create Agent and Buffer, then run training
run' :: Params -> CircusUrl -> Tracker -> String -> Mode -> Algorithm 
     -> ReplayMemory -> IO ()
run' p@Params{..} addr tracker path Train TD3 HER = do
    actDim     <- actionSpace addr
    (od,gd,_)  <- observationSpace addr
    let obsDim =  od + gd
        buf    =  HER.empty

    TD3.mkAgent p obsDim actDim >>= train p addr tracker path numEpisodes buf
run' p addr tracker path Eval  TD3 HER = do
    actDim     <- actionSpace addr
    (od,gd,_)  <- observationSpace addr
    let obsDim =  od + gd
    agent      <- ALG.loadAgent p path obsDim actDim 0 :: IO TD3.Agent
    (state,_,goal) <- reset addr
    let obs      = T.cat (T.Dim 1) [state, goal]
        success' = T.full [head $ T.shape obs] False 
                 $ T.withDType T.Bool T.defaultOpts
    trackEnvState tracker addr 0
    eval p addr tracker agent 0 0 obs success'
    pure()
run' _ _    _       _    _     _   _  = error "Not Implemented"

-- | Clown School
run :: Args -> IO ()
run Args{..} = do
    params <- parseConfig config
    
    T.manualSeed $ rngSeed params

    path'   <- if mode' == Train 
                  then createModelArchiveDir' path algorithm cktId 
                                              cktBackend "0" cktSpace
                  else pure path

    let url' = url cktHost cktPort cktId cktBackend cktSpace "0"
        uri' = trackingURI mlfHost mlfPort
        exp' = algorithm ++ "-" ++ buffer ++ "-" ++ cktId ++ "-" ++ cktBackend
                         ++ "-" ++ cktSpace ++ "-v" ++ var ++ "-" ++ mode
    
    nEnvs   <- numEnvs url'
    tracker <- mkTracker uri' exp' >>= newRuns' nEnvs

    run' params url' tracker path' mode' alg' buf'
    endRuns' tracker
  where
    mode' = read mode      :: Mode
    alg'  = read algorithm :: Algorithm
    buf'  = read buffer    :: ReplayMemory
    var   = "0"
