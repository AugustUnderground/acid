{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}

-- | Twin Delayed Deep Deterministic Policy Gradient Algorithm
module ALG.TD3 ( PolicyNet (..)
               , CriticNet (..)
               , Agent (..)
               , mkAgent
               , saveAgent'
               ) where

import           Lib
import           ALG.HyperParameters
import qualified ALG
import           MLFlow.Extensions
import           CKT                              hiding (url)
import           Control.Monad
import           GHC.Generics
import qualified Torch                     as T
import qualified Torch.Extensions          as T
import qualified Torch.Functional.Internal as T          (negative)
import qualified Torch.NN                  as NN

------------------------------------------------------------------------------
-- Neural Networks
------------------------------------------------------------------------------

-- | Policy Network Specification
data PolicyNetSpec = PolicyNetSpec Int Int
    deriving (Show, Eq)

-- | Critic Network Specification
data CriticNetSpec = CriticNetSpec Int Int
    deriving (Show, Eq)

-- | Actor Network Architecture
data PolicyNet = PolicyNet { pLayer0 :: T.Linear
                           , pLayer1 :: T.Linear
                           , pLayer2 :: T.Linear 
                           } deriving (Generic, Show, T.Parameterized)

-- | Critic Network Architecture
data CriticNet = CriticNet { q1Layer0 :: T.Linear
                           , q1Layer1 :: T.Linear
                           , q1Layer2 :: T.Linear 
                           , q2Layer0 :: T.Linear
                           , q2Layer1 :: T.Linear
                           , q2Layer2 :: T.Linear 
                           } deriving (Generic, Show, T.Parameterized)

-- | Actor Network Weight initialization
instance T.Randomizable PolicyNetSpec PolicyNet where
    sample (PolicyNetSpec obsDim actDim) = 
        PolicyNet <$> T.sample   (T.LinearSpec obsDim hidDim) 
                  <*> T.sample   (T.LinearSpec hidDim hidDim)
                  <*> ( T.sample (T.LinearSpec hidDim actDim)
                                    >>= weightInitUniform (- wInit) wInit )

-- | Critic Network Weight initialization
instance T.Randomizable CriticNetSpec CriticNet where
    sample (CriticNetSpec obsDim actDim) = 
        CriticNet <$> T.sample   (T.LinearSpec dim    hidDim) 
                  <*> T.sample   (T.LinearSpec hidDim hidDim) 
                  <*> ( T.sample (T.LinearSpec hidDim 1) 
                        >>= weightInitUniform (- wInit) wInit )
                  <*> T.sample   (T.LinearSpec dim    hidDim) 
                  <*> T.sample   (T.LinearSpec hidDim hidDim) 
                  <*> ( T.sample (T.LinearSpec hidDim 1) 
                        >>= weightInitUniform (- wInit) wInit )
        where dim = obsDim + actDim

-- | Policy Network Forward Pass
π :: PolicyNet -> T.Tensor -> T.Tensor
π PolicyNet{..} o = a
  where
    a = T.tanh . T.linear pLayer2
      . T.relu . T.linear pLayer1
      . T.relu . T.linear pLayer0
      $ o

-- | Critic Network Forward Pass
q :: CriticNet -> T.Tensor -> T.Tensor -> (T.Tensor, T.Tensor)
q CriticNet{..} o a = (v1,v2)
  where 
    x  = T.cat (T.Dim $ -1) [o,a]
    v1 = T.linear q1Layer2 . T.relu
       . T.linear q1Layer1 . T.relu
       . T.linear q1Layer0 $ x
    v2 = T.linear q2Layer2 . T.relu
       . T.linear q2Layer1 . T.relu
       . T.linear q2Layer0 $ x

-- | Convenience Function, takes the minimum of both online actors
q' :: CriticNet -> T.Tensor -> T.Tensor -> T.Tensor
q' cn o a = fst . T.minDim (T.Dim 1) T.KeepDim . T.cat (T.Dim 1) $ [v1,v2]
  where
    (v1,v2) = q cn o a

-------------------------------------------------------------------------------
-- Twin Delayed Deep Deterministic Policy Gradient Agent
------------------------------------------------------------------------------

-- | TD3 Agent
data Agent = Agent { φ      :: PolicyNet   -- ^ Online Policy φ
                   , φ'     :: PolicyNet   -- ^ Target Policy φ'
                   , θ      :: CriticNet   -- ^ Online Critic θ
                   , θ'     :: CriticNet   -- ^ Target Critic θ
                   , φOptim :: T.Adam      -- ^ Policy Optimizer
                   , θOptim :: T.Adam      -- ^ Critic Optimizer
                   } deriving (Generic, Show)

-- | TD3 Agent is an implements Agent
instance ALG.Agent Agent where
  saveAgent      = saveAgent
  loadAgent      = loadAgent
  act            = act
  act'           = act'
  act''          = act''
  updatePolicy   = updatePolicy

-- | Agent constructor
mkAgent :: Int -> Int -> IO Agent
mkAgent obsDim actDim = do
    φOnline  <- T.toFloat <$> T.sample (PolicyNetSpec obsDim actDim)
    φTarget' <- T.toFloat <$> T.sample (PolicyNetSpec obsDim actDim)
    θOnline  <- T.toFloat <$> T.sample (CriticNetSpec obsDim actDim)
    θTarget' <- T.toFloat <$> T.sample (CriticNetSpec obsDim actDim)

    let φTarget = copySync φTarget' φOnline
        θTarget = copySync θTarget' θOnline
        φOpt    = T.mkAdam 0 β1 β2 (NN.flattenParameters φOnline)
        θOpt    = T.mkAdam 0 β1 β2 (NN.flattenParameters θOnline)

    pure $ Agent φOnline φTarget θOnline θTarget φOpt θOpt

-- | Save an Agent Checkpoint
saveAgent :: FilePath -> Agent -> IO Agent
saveAgent path agent@Agent{..} = do
    T.saveParams φ  (path ++ "/actorOnline.pt")
    T.saveParams φ' (path ++ "/actorTarget.pt")
    T.saveParams θ  (path ++ "/criticOnline.pt")
    T.saveParams θ' (path ++ "/criticTarget.pt")

    saveOptim φOptim (path ++ "/actorOptim")
    saveOptim θOptim (path ++ "/criticOptim")

    putStrLn $ "\tSaving Checkpoint at " ++ path ++ " ... "

    pure agent

-- | Save an Agent and return the agent
saveAgent' :: FilePath -> Agent -> IO ()
saveAgent' p a = void $ saveAgent p a

-- | Load an Agent Checkpoint
loadAgent :: String -> Int -> Int -> Int -> IO Agent
loadAgent path obsDim actDim iter = do
        Agent{..} <- mkAgent obsDim actDim

        fφ    <- T.loadParams φ       (path ++ "/actorOnline.pt")
        fφ'   <- T.loadParams φ'      (path ++ "/actorTarget.pt")
        fθ    <- T.loadParams θ       (path ++ "/criticOnline.pt")
        fθ'   <- T.loadParams θ'      (path ++ "/criticTarget.pt")
        fφOpt <- loadOptim iter β1 β2 (path ++ "/actorOptim")
        fθOpt <- loadOptim iter β1 β2 (path ++ "/criticOptim")
       
        pure $ Agent fφ fφ' fθ fθ' fφOpt fθOpt

-- | Select an action from target policy with clipped noise
act :: Agent -> T.Tensor -> IO T.Tensor
act Agent{..} s = do
    ε' <- T.toFloat <$> T.randnLikeIO a 
    let ε = T.clamp (- c) c (ε' * σEval)
    pure $ T.clamp actionLow actionHigh (a + ε)
  where
    a = π φ' s

-- | Select action from online policy with Exploration Noise
act' :: Agent -> T.Tensor -> IO T.Tensor
act' Agent{..} s = do
    ε <- T.toFloat <$> T.randnLikeIO a
    pure . T.toDevice d' $ T.clamp actionLow actionHigh (a + (ε * σAct))
  where
    d' = T.device s
    s' = T.toDevice T.gpu s
    a = π φ s'

-- | Select an action from online policy without any noise
act'' :: Agent -> T.Tensor -> T.Tensor
act'' Agent{..} = T.toDevice T.cpu . π φ . T.toDevice T.gpu

-- | Policy Update Step
updateStep :: Int -> Int -> Agent -> Tracker -> Transition -> IO Agent
updateStep iter epoch agent@Agent{..} tracker (s,a,r,s',d') = do
    a' <- act agent s' >>= T.detach
    v' <- T.detach . T.squeezeAll $ q' θ' s' a'
    y  <- T.detach $ r + ((1.0 - d') * γ * v')

    let (v1, v2) = both T.squeezeAll $ q θ s a
        jQ       = T.mseLoss v1 y + T.mseLoss v2 y

    (θOnline', θOptim') <- T.runStep θ θOptim jQ ηθ

    when (verbose && epoch `mod` 10 == 0) do
        putStrLn $ "\tEpoch " ++ show epoch ++ ":"
        putStrLn $ "\t\tΘ Loss:\t" ++ show jQ

    _ <- trackLoss tracker (iter' !! epoch) "Critic_Loss" (T.asValue jQ :: Float)

    (φOnline', φOptim')  <- if epoch `mod` d == 0 
                               then updateActor
                               else pure (φ, φOptim)

    (φTarget', θTarget') <- if epoch == (numEpochs - 1)
                               then syncTargets
                               else pure (φ', θ')

    pure $ Agent φOnline' φTarget' θOnline' θTarget' φOptim' θOptim'
  where
    iter' = map ((iter * numEpochs) +) $ range numEpochs
    updateActor :: IO (PolicyNet, T.Adam)
    updateActor = do
        when (verbose && epoch `mod` 10 == 0) do
            putStrLn $ "\t\tφ Loss:\t" ++ show jφ
        _ <- trackLoss tracker ((iter' !! epoch) `div` d)
                       "Actor_Loss" (T.asValue jφ :: Float)
        T.runStep φ φOptim jφ ηφ
      where
        (v,_) = q θ s $ π φ s
        jφ    = T.negative . T.mean $ v
    syncTargets :: IO (PolicyNet, CriticNet)
    syncTargets = do
        when verbose do
            putStrLn "\t\tUpdating Targets."
        φTarget' <- softSync τ φ' φ
        θTarget' <- softSync τ θ' θ
        pure (φTarget', θTarget')

-- | Update TD3 Policy
updatePolicy :: CircusUrl -> Tracker -> Int -> [Transition] -> Agent -> IO Agent
updatePolicy _   _       _    []              agent = pure agent
updatePolicy url tracker iter (batch:batches) agent =
    updateStep iter epoch agent tracker batch >>= 
        updatePolicy url tracker iter batches
  where
    epoch = numEpochs - length batches - 1
