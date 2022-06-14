{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Circuis REST API Communication
module CKT ( Circuit (..)
           , PDK (..)
           , Space (..)
           , CircusUrl
           , url
           , numEnvs
           , actionSpace
           , observationSpace
           , reset
           , reset'
           , step
           , randomAction
           , calculateReward
           , Observation (..)
           , Info (..)
           , Action (..)
           , currentPerformance
           , currentGoal
           , currentSizing
           , lastAction
           ) where

import           Lib
import           Data.Aeson
import qualified Data.Map              as M
import           Data.Maybe                           (fromJust)
import           Control.Lens                  hiding ((.=))
import           GHC.Generics                  hiding (Meta)
import           Network.Wreq          as Wreq hiding (get, post)
import           Network.HTTP.Client   as HTTP
import qualified Data.ByteString.Lazy  as BL
import qualified Data.ByteString       as BS   hiding (pack)
import qualified Torch                 as T

------------------------------------------------------------------------------
-- Circus Server Interaction
------------------------------------------------------------------------------

-- | Available Circuits
data Circuit = OP1 -- ^ Miller Amplifier (op1)
             | OP2 -- ^ Symmetrical Amplifier (op2)
             | OP8 -- ^ Folded Cascode (op8)
    deriving (Eq)

instance Show Circuit where
  show OP1 = "op1"
  show OP2 = "op2"
  show OP8 = "op8"

instance Read Circuit where
  readsPrec _ "op1" = [(OP1, "")]
  readsPrec _ "op2" = [(OP2, "")]
  readsPrec _ "op8" = [(OP8, "")]
  readsPrec _ _     = undefined

-- | Available PDKs
data PDK = XH035 -- ^ X-Fab XH035 350nm Process
         | XH018 -- ^ X-Fab XH018 180nm Process
    deriving (Eq)

instance Show PDK where
  show XH035 = "xh035"
  show XH018 = "xh018"

instance Read PDK where
  readsPrec _ "xh035" = [(XH035, "")]
  readsPrec _ "xh018" = [(XH018, "")]
  readsPrec _ _       = undefined

-- | Available Design / Action Spaces
data Space = Electric  -- ^ Electric Design Space
           | Geometric -- ^ Geometric Design Space
    deriving (Eq)

instance Show Space where
  show Electric  = "elec"
  show Geometric = "geom"

instance Read Space where
  readsPrec _ "elec" = [(Electric, "")]
  readsPrec _ "geom" = [(Geometric, "")]
  readsPrec _ _      = undefined

-- | HTTP options for Circuit server communication, sometimes simulations can take
-- a while, therefore we wait ...
httpOptions :: Wreq.Options
httpOptions = Wreq.defaults & manager 
           .~ Left ( HTTP.defaultManagerSettings 
                        { HTTP.managerResponseTimeout = 
                            HTTP.responseTimeoutNone } )

-- | Info Dict returned from Environment
data Info = Info { goal    :: ![String] -- ^ Goal Parameters
                 , inputs  :: ![String] -- ^ Parameters in Action Vector
                 , outputs :: ![String] -- ^ Parameters in State Vector
                 } deriving (Generic, Show)

instance FromJSON Info
instance ToJSON   Info

-- | Action passed to Environment
newtype Action a = Action { action :: a
                        } deriving (Generic, Show)

instance FromJSON (Action [[Float]])
instance ToJSON   (Action [[Float]])

instance Functor Action where
  fmap f (Action a) = Action (f a)

-- | Observation returned from Stepping / Resetting Environment
data Observation a = Observation { observation  :: !a            -- ^ State
                                 , achievedGoal :: !a            -- ^ Achieved Goal
                                 , desiredGoal  :: !a            -- ^ Desired Goal
                                 , done         :: Maybe [Bool]  -- ^ Terminal Flag
                                 , reward       :: Maybe [Float] -- ^ Rewards
                                 , info         :: Maybe [Info]  -- ^ Info Dict
                                 } deriving (Generic, Show)

instance FromJSON (Observation [[Float]]) where
  parseJSON (Object v) = do
        observation'  <- v .:  "observation"
        achievedGoal' <- v .:  "achieved_goal"
        desiredGoal'  <- v .:  "desired_goal"
        done'         <- v .:? "done"
        reward'       <- v .:? "reward"
        info'         <- v .:? "info"
        pure          (Observation observation' achievedGoal' desiredGoal' 
                                   done' reward' info')
  parseJSON _ = fail "Expected an Object"

instance ToJSON (Observation [[Float]]) where
  toJSON Observation{..} = object [ "observation"   .= observation
                                  , "achieved_goal" .= achievedGoal
                                  , "desired_goal"  .= desiredGoal
                                  , "done"          .= done
                                  , "reward"        .= reward
                                  , "info"          .= info ]

instance Functor Observation where
  fmap f (Observation o a g d r i) = Observation (f o) (f a) (f g) d r i

-- | Meta information obtained from Circus Server
type Meta = M.Map String Int

-- | Base Route to Cricus Server
type CircusUrl = String

-- | Generate URL to a Circuit server from meta information
url :: String -> String -> String -> String -> String -> String -> CircusUrl
url h p i b s v = "http://" ++ h ++ ":" ++ p ++ "/" ++ i ++ "-" ++ b 
                                 ++ "-" ++ s ++ "-v" ++ v

-- | Send a HTTP GET Request to Circus Server
get :: CircusUrl -> String -> IO BS.ByteString
get addr route =  BL.toStrict . (^. Wreq.responseBody) 
              <$> getWith httpOptions (addr ++ "/" ++ route)

-- | Send a HTTP POST Request to a Circus Server
post :: CircusUrl -> String -> Value -> IO BS.ByteString
post addr route payload =  BL.toStrict . (^. Wreq.responseBody) 
                       <$> postWith httpOptions (addr ++ "/" ++ route) payload 

-- | lookup a list of fields from meta response
meta :: CircusUrl -> String -> [String] -> IO [Int]
meta addr route fields =  fromJust . lookup' fields . fromJust 
                      <$> (decodeStrict <$> get addr route :: IO (Maybe Meta))

-- | Get the number of Environments from the given Circus Server instance
numEnvs :: CircusUrl -> IO Int
numEnvs addr = head <$> meta addr "num_envs" ["num"]

-- | Get the number of Environments from the given Circus Server instance
actionSpace :: CircusUrl -> IO Int
actionSpace addr = head <$> meta addr "action_space" ["action"]

-- | Get the number of Environments from the given Circus Server instance
observationSpace :: CircusUrl -> IO (Int, Int, Int)
observationSpace addr = do
    os:ag:dg:_ <- meta addr "observation_space" fields
    pure (os,ag,dg)
  where
    fields = ["observation", "achieved_goal", "desired_goal"]

-- | Restart Environment and Restore last state
recoverLast :: CircusUrl -> IO (Observation [[Float]])
recoverLast addr = fromJust . decodeStrict <$> get addr "restore_last"

-- | Reset Environment at given URL
reset'' :: CircusUrl -> [Bool] -> IO (Observation [[Float]])
reset'' addr  []  = fromJust . decodeStrict <$> get  addr "reset"
reset'' addr mask = fromJust . decodeStrict <$> post addr "reset" msk
  where
    msk = toJSON (M.fromList [("env_mask", mask)] :: (M.Map String [Bool]))

-- | Reset Environments and get (observation, achieved_goal, desired_goal)
reset :: CircusUrl -> IO (T.Tensor, T.Tensor, T.Tensor)
reset addr = do
    (Observation observation achieved desired _ _ _) 
            <- fmap T.asTensor <$> reset'' addr []
    pure (observation, achieved, desired)

-- | Reset subset of environments with given mask
reset' :: CircusUrl -> T.Tensor -> IO (T.Tensor, T.Tensor, T.Tensor)
reset' addr msk = do
    obs <- fmap T.asTensor <$> reset'' addr msk'
    pure (observation obs, achievedGoal obs, desiredGoal obs)
  where
    msk' = T.asValue . T.squeezeAll $ msk :: [Bool]

-- | Get a random action sampled from the environments action space
randomAction' :: CircusUrl -> IO (Action [[Float]])
randomAction' addr = fromJust . decodeStrict <$> get addr "random_action"

-- | Shorthand for getting Tensors
randomAction :: CircusUrl -> IO T.Tensor
randomAction addr = action . fmap T.asTensor <$> randomAction' addr

-- | Recover Last state
recover :: CircusUrl -> Maybe (Observation [[Float]]) 
        -> IO (Observation [[Float]])
recover _    (Just obs) = pure obs
recover addr Nothing    = recoverLast addr

-- | Take an action in a given Environment and get the new observation
step' :: CircusUrl -> Action [[Float]] -> IO (Observation [[Float]])
step' addr action' = post addr "step" (toJSON action') 
                        >>= recover addr . decodeStrict

-- | Shorthand for taking a Tensor action and returning Tensors
step :: CircusUrl -> T.Tensor 
     -> IO (T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor)
step addr action' = do
    obs <- fmap T.asTensor <$> step' addr action''
    pure ( observation obs, achievedGoal obs, desiredGoal obs
         , T.asTensor . fromJust . reward $ obs
         , T.asTensor . fromJust . done   $ obs )
  where
    action'' = Action (T.asValue action' :: [[Float]])

-- | Calcualte reward for a given achieved and desired goal
calculateReward' :: CircusUrl -> Observation [[Float]] -> IO (M.Map String [Float])
calculateReward' addr obs = fromJust . decodeStrict <$> post addr "reward" obs'
  where 
    obs' = toJSON obs

-- | Shorthand for calculating reward given Tensors
calculateReward :: CircusUrl -> T.Tensor -> T.Tensor -> IO T.Tensor
calculateReward addr ag dg = do
    rew <- fromJust . M.lookup "reward" <$> calculateReward' addr obs
    pure $ T.asTensor (rew :: [Float])
  where 
    obs = Observation [[]] (T.asValue ag :: [[Float]]) 
                           (T.asValue dg :: [[Float]]) 
                           Nothing Nothing Nothing

-- | Get current State of the Environment
currentState :: CircusUrl -> String -> IO (M.Map Int (M.Map String Float))
currentState addr route = fromJust . decodeStrict <$> get addr route

-- | Shorthand for getting Performance
currentPerformance :: CircusUrl -> IO (M.Map Int (M.Map String Float))
currentPerformance addr = currentState addr "current_performance"

-- | Shorthand for getting Goal
currentGoal :: CircusUrl -> IO (M.Map Int (M.Map String Float))
currentGoal addr = currentState addr "current_goal"

-- | Shorthand for getting Sizing
currentSizing :: CircusUrl -> IO (M.Map Int (M.Map String Float))
currentSizing addr = currentState addr "current_sizing"

-- | Shorthand for getting last Action
lastAction :: CircusUrl -> IO (M.Map Int (M.Map String Float))
lastAction addr = currentState addr "last_action"
