{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE BangPatterns    #-}
{-# LANGUAGE BlockArguments  #-}
{-# LANGUAGE RecordWildCards #-}

-- | Artificial Circuit Designer
module ACiD where

import           Control.Monad
import           Lib
import           CKT
import           CFG
import           MLFlow
import           MLFlow.Extensions
import           ALG
import           RPB
import qualified ALG.TD3             as TD3
import qualified RPB.HER             as HER
import qualified Torch               as T
import qualified Torch.Extensions    as T

-- | Policy evaluation Episode
eval :: (Agent a) => Meta -> HyperParameters -> CircusUrl -> Tracker -> a 
     -> Int -> Int -> T.Tensor -> T.Tensor -> IO ()
eval meta@Meta{..} hp addr tracker agent episode t obs dones | T.all dones = pure ()
                                                             | otherwise   = do
    (!state',_,!goal,!reward,!done) <- step addr $ act'' agent obs

    let dones'  = T.logicalOr done dones
        obs'    = T.cat (T.Dim 1) [state', goal]
        success = successRate . T.logicalOr dones . T.logicalAnd done $ T.ge reward 0.0
    
    when (verbose && t % 10 == 0) do
        putStrLn $ "\tStep " ++ show t ++ ":"
        putStrLn $ "\t\tSuccess Rate: " ++ show success ++ "%"

    _ <- trackLoss tracker (iter' !! t) "Success" success

    eval meta hp addr tracker agent episode t' obs' dones'
  where
    t'    = t + 1
    iter' = map ((episode * horizonT) +) [0..]

-- | Runs training on given Agent with Buffer
train :: (Agent a, ReplayBuffer b) => Meta -> HyperParameters -> CircusUrl 
      -> Tracker -> String -> Int -> b T.Tensor -> a -> IO ()
train _ _ _    _       path 0       _      agent = void $ saveAgent path agent
train meta@Meta{..} hp@HyperParameters{..} addr tracker path episode buf agent = do

    when verbose do
        let isRandom = if iter % explFreq == 0 
                          then "Random Exploration" 
                          else "Policy Exploitation"
        putStrLn $ "Episode " ++ show iter ++ " (" ++ isRandom ++ "):"

    buf'  <-  push bufferSize buf 
             <$> collectExperience meta hp addr tracker iter agent

    batches  <-  map (tmap (T.toDevice T.gpu)) 
             <$> randomBatches numEpochs batchSize buf'

    agent'   <-  updatePolicy meta hp addr tracker iter batches agent 
                    >>= saveAgent path

    when (iter /= 0 && iter % evalFreq == 0) do
        (state,_,goal) <- reset addr
        let obs      = T.cat (T.Dim 1) [state, goal]
            iter'    = iter // evalFreq
            success' = T.full [head $ T.shape obs] False 
                     $ T.withDType T.Bool T.defaultOpts
        putStrLn $ "Policy Evaluation Episode " ++ show iter'
        eval meta hp addr tracker agent' iter' 0 obs success'
        pure ()

    train meta hp addr tracker path episode' buf' agent'
  where
    episode' = episode - 1
    iter     = numEpisodes - episode

-- | Create Agent and Buffer, then run training
run' :: Meta -> HyperParameters -> CircusUrl -> Tracker -> String -> Mode 
     -> Algorithm -> ReplayMemory -> IO ()
run' meta@Meta{..} hp addr tracker path Train TD3 HER = do
    actDim     <- actionSpace addr
    (od,gd,_)  <- observationSpace addr
    let obsDim =  od + gd
        buf    =  HER.empty

    TD3.mkAgent hp obsDim actDim >>= train meta hp addr tracker path numEpisodes buf
run' meta hp addr tracker path Eval TD3 HER = do
    actDim     <- actionSpace addr
    (od,gd,_)  <- observationSpace addr
    let obsDim =  od + gd
    agent      <- ALG.loadAgent hp path obsDim actDim 0 :: IO TD3.Agent
    (state,_,goal) <- reset addr
    let obs      = T.cat (T.Dim 1) [state, goal]
        success' = T.full [head $ T.shape obs] False 
                 $ T.withDType T.Bool T.defaultOpts
    eval meta hp addr tracker agent 0 0 obs success'
    pure()
run' _    _  _    _       _    _     _   _  = error "Not Implemented"

-- | Clown School
run :: Args -> IO ()
run Args{..} = do
    config' <- parseConfig config

    let alg = show . algorithm  . meta $ config'
        buf = show . buffer     . meta $ config'
        ace = show . aceId      . meta $ config'
        pdk = show . aceBackend . meta $ config'
        var = show . variant    . meta $ config'
        spc = show . space      . meta $ config'

    path'   <- if mode' == Train 
                  then createModelArchiveDir' path alg ace pdk var spc
                  else pure path

    let url' = url cktHost cktPort ace pdk spc var
        uri' = trackingURI mlfHost mlfPort
        alg' = algorithm . meta $ config'
        buf' = buffer    . meta $ config'
        exp' = alg ++ "-" ++ buf ++ "-" ++ ace ++ "-" ++ pdk
                   ++ "-" ++ spc ++ "-v" ++ var ++ "-" ++ mode
    
    nEnvs   <- numEnvs url'
    tracker <- mkTracker uri' exp' >>= newRuns' nEnvs

    let meta' = meta config'
        hp    = hyperParameters config'
    
    run' meta' hp url' tracker path' mode' alg' buf'
  where
    mode'   = read mode :: Mode
