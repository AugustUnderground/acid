{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE BlockArguments        #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE RecordWildCards       #-}
{-# LANGUAGE MultiParamTypeClasses #-}

-- | Twin Delayed Deep Deterministic Policy Gradient Algorithm
module ALG.TD3 ( PolicyNet (..)
               , CriticNet (..)
               , Agent (..)
               , mkAgent
               , saveAgent'
               ) where

import           Lib
import qualified ALG
import           MLFlow.Extensions
import           CKT                              hiding (url)
import           HyperParameters
import           Control.Monad
import           GHC.Generics                     hiding (Meta)
import qualified Torch                     as T
import qualified Torch.Extensions          as T
import qualified Torch.Functional.Internal as T          (negative)
import qualified Torch.NN                  as NN

------------------------------------------------------------------------------
-- Neural Networks
------------------------------------------------------------------------------

-- | Policy Network Specification
data PolicyNetSpec = PolicyNetSpec Int Int Int Float
    deriving (Show, Eq)

-- | Critic Network Specification
data CriticNetSpec = CriticNetSpec Int Int Int Float
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
    sample (PolicyNetSpec obsDim actDim hidDim' wInit') = 
        PolicyNet <$> T.sample   (T.LinearSpec obsDim  hidDim') 
                  <*> T.sample   (T.LinearSpec hidDim' hidDim')
                  <*> ( T.sample (T.LinearSpec hidDim' actDim)
                                    >>= weightInitUniform (- wInit') wInit' )

-- | Critic Network Weight initialization
instance T.Randomizable CriticNetSpec CriticNet where
    sample (CriticNetSpec obsDim actDim hidDim' wInit') = 
        CriticNet <$> T.sample   (T.LinearSpec dim     hidDim') 
                  <*> T.sample   (T.LinearSpec hidDim' hidDim') 
                  <*> ( T.sample (T.LinearSpec hidDim' 1) 
                        >>= weightInitUniform (- wInit') wInit' )
                  <*> T.sample   (T.LinearSpec dim     hidDim') 
                  <*> T.sample   (T.LinearSpec hidDim' hidDim') 
                  <*> ( T.sample (T.LinearSpec hidDim' 1) 
                        >>= weightInitUniform (- wInit') wInit' )
        where dim = obsDim + actDim

-- | Policy Network Forward Pass
?? :: PolicyNet -> T.Tensor -> T.Tensor
?? PolicyNet{..} o = a
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
data Agent = Agent { ??      :: PolicyNet   -- ^ Online Policy ??
                   , ??'     :: PolicyNet   -- ^ Target Policy ??'
                   , ??      :: CriticNet   -- ^ Online Critic ??
                   , ??'     :: CriticNet   -- ^ Target Critic ??'
                   , ??Optim :: T.Adam      -- ^ Policy Optimizer
                   , ??Optim :: T.Adam      -- ^ Critic Optimizer
                   , actLo  :: Float       -- ^ Lower bound of Action space
                   , actHi  :: Float       -- ^ Upper bound of Action space
                   , ??A     :: T.Tensor    -- ^ Action Noise
                   , ??E     :: T.Tensor    -- ^ Eval Noise
                   , c'     :: Float       -- ^ Noise Clipping
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
mkAgent :: Params -> Int -> Int -> IO Agent
mkAgent Params{..} obsDim actDim = do
    ??Online  <- T.toFloat <$> T.sample (PolicyNetSpec obsDim actDim hidDim wInit)
    ??Target' <- T.toFloat <$> T.sample (PolicyNetSpec obsDim actDim hidDim wInit)
    ??Online  <- T.toFloat <$> T.sample (CriticNetSpec obsDim actDim hidDim wInit)
    ??Target' <- T.toFloat <$> T.sample (CriticNetSpec obsDim actDim hidDim wInit)

    let ??Target = copySync ??Target' ??Online
        ??Target = copySync ??Target' ??Online
        ??Opt    = T.mkAdam 0 ??1 ??2 (NN.flattenParameters ??Online)
        ??Opt    = T.mkAdam 0 ??1 ??2 (NN.flattenParameters ??Online)
        ??Act'   = T.toDevice T.gpu ??Act
        ??Eval'  = T.toDevice T.gpu ??Eval

    pure $ Agent ??Online ??Target ??Online ??Target ??Opt ??Opt 
                 actionLow actionHigh ??Act' ??Eval' c

-- | Save an Agent Checkpoint
saveAgent :: FilePath -> Agent -> IO Agent
saveAgent path agent@Agent{..} = do
    T.saveParams ??  (path ++ "/actorOnline.pt")
    T.saveParams ??' (path ++ "/actorTarget.pt")
    T.saveParams ??  (path ++ "/criticOnline.pt")
    T.saveParams ??' (path ++ "/criticTarget.pt")

    saveOptim ??Optim (path ++ "/actorOptim")
    saveOptim ??Optim (path ++ "/criticOptim")

    putStrLn $ "\tSaving Checkpoint at " ++ path ++ " ... "

    pure agent

-- | Save an Agent and return the agent
saveAgent' :: FilePath -> Agent -> IO ()
saveAgent' p a = void $ saveAgent p a

-- | Load an Agent Checkpoint
loadAgent :: Params -> String -> Int -> Int -> Int -> IO Agent
loadAgent p@Params{..} path obsDim actDim iter = do
        Agent{..} <- mkAgent p obsDim actDim

        f??    <- T.loadParams ??       (path ++ "/actorOnline.pt")
        f??'   <- T.loadParams ??'      (path ++ "/actorTarget.pt")
        f??    <- T.loadParams ??       (path ++ "/criticOnline.pt")
        f??'   <- T.loadParams ??'      (path ++ "/criticTarget.pt")
        f??Opt <- loadOptim iter ??1 ??2 (path ++ "/actorOptim")
        f??Opt <- loadOptim iter ??1 ??2 (path ++ "/criticOptim")
       
        pure $ Agent f?? f??' f?? f??' f??Opt f??Opt actionLow actionHigh ??Act ??Eval c

-- | Select an action from target policy with clipped noise
act :: Agent -> T.Tensor -> IO T.Tensor
act Agent{..} s = do
    ??' <- T.toFloat <$> T.randnLikeIO a 
    let ?? = T.clamp (- c') c' (??' * ??E)
    pure $ T.clamp actLo actHi (a + ??)
  where
    a = ?? ??' s

-- | Select action from online policy with Exploration Noise
act' :: Agent -> T.Tensor -> IO T.Tensor
act' Agent{..} s = do
    ?? <- T.toFloat <$> T.randnLikeIO a
    pure . T.toDevice d' $ T.clamp actLo actHi (a + (?? * ??A))
  where
    d' = T.device s
    s' = T.toDevice T.gpu s
    a = ?? ?? s'

-- | Select an action from online policy without any noise
act'' :: Agent -> T.Tensor -> T.Tensor
act'' Agent{..} s = T.toDevice dev' . ?? ?? . T.toDevice dev $ s
  where
    dev' = T.device s
    dev  = T.gpu

-- | Policy Update Step
updateStep :: Params -> Int -> Int -> Agent -> Tracker -> Transition -> IO Agent
updateStep Params{..} iter epoch agent@Agent{..} tracker trans = do
    a' <- act agent s' >>= T.detach
    v' <- T.detach . T.squeezeAll $ q' ??' s' a'
    y  <- T.detach $ r + ((1.0 - d') * ?? * v')

    let (v1, v2) = both T.squeezeAll $ q ?? s a
        jQ       = T.mseLoss v1 y + T.mseLoss v2 y

    (??Online', ??Optim') <- T.runStep ?? ??Optim jQ ????

    when (epoch % 10 == 0) do
        putStrLn $ "\tEpoch " ++ show epoch ++ ":"
        putStrLn $ "\t\t?? Loss:\t" ++ show jQ

    _ <- trackLoss tracker (iter' !! epoch) "Critic_Loss" (T.asValue jQ :: Float)

    (??Online', ??Optim')  <- if epoch % d == 0 
                               then updateActor
                               else pure (??, ??Optim)

    (??Target', ??Target') <- if epoch == (numEpochs - 1)
                               then syncTargets
                               else pure (??', ??')

    pure $ Agent ??Online' ??Target' ??Online' ??Target' ??Optim' ??Optim'
                 actLo actHi ??A ??E c'

  where
    (s,a,r,s',d') = trans
    iter' = map ((iter * numEpochs) +) $ range numEpochs
    updateActor :: IO (PolicyNet, T.Adam)
    updateActor = do
        when (epoch % 10 == 0) do
            putStrLn $ "\t\t?? Loss:\t" ++ show j??
        _ <- trackLoss tracker ((iter' !! epoch) // d)
                       "Actor_Loss" (T.asValue j?? :: Float)
        T.runStep ?? ??Optim j?? ????
      where
        (v,_) = q ?? s $ ?? ?? s
        j??    = T.negative . T.mean $ v
    syncTargets :: IO (PolicyNet, CriticNet)
    syncTargets = do
        putStrLn "\t\tUpdating Targets."
        ??Target' <- softSync ?? ??' ??
        ??Target' <- softSync ?? ??' ??
        pure (??Target', ??Target')

-- | Update TD3 Policy
updatePolicy :: Params -> CircusUrl -> Tracker -> Int -> [Transition] -> Agent 
             -> IO Agent
updatePolicy _ _ _ _ [] agent = pure agent
updatePolicy p@Params{..} url tracker iter (batch:batches) agent =
    updateStep p iter epoch agent tracker batch >>= 
        updatePolicy p url tracker iter batches
  where
    epoch = numEpochs - length batches - 1
