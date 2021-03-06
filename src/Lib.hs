{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE DeriveGeneric       #-}
{-# LANGUAGE DeriveAnyClass      #-}
{-# LANGUAGE BlockArguments      #-}
{-# LANGUAGE RecordWildCards     #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE TypeApplications    #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- | Utility and Helper functions for EDELWACE
module Lib where

import           GHC.Generics
import qualified Data.Map           as M
import qualified Data.Set           as S
import           Data.Aeson
import           Data.Time.Clock          (getCurrentTime)
import           Data.Time.Format         (formatTime, defaultTimeLocale)
import           System.Directory
import qualified Torch              as T
import qualified Torch.NN           as NN
import qualified Torch.Initializers as T
import qualified Torch.Extensions   as T

------------------------------------------------------------------------------
-- Types and Aliases
------------------------------------------------------------------------------

-- | Available Algorithms
data Algorithm = TD3 -- ^ Twin Delayed Deep Deterministic Policy Gradient
               | SAC -- ^ Soft Actor Critic
               | PPO -- ^ Proximal Policy Optimization
               deriving (Eq, Show, Read, Generic, FromJSON, ToJSON)

-- | Available Replay Buffer Types
data ReplayMemory = RPB -- ^ Vanilla Replay Buffer
                  | PER -- ^ Prioritized Experience Replay
                  | MEM -- ^ PPO Style replay Memory
                  | ERE -- ^ Emphasizing Recent Experience
                  | HER -- ^ Hindsight Experience Replay
                  deriving (Eq, Show, Read, Generic, FromJSON, ToJSON)

-- | Hindsight Experience Replay Strategies for choosing Goals
data Strategy = Final   -- ^ Only Final States are additional targets
              | Random  -- ^ Replay with `k` random states encountered so far (basically vanilla)
              | Episode -- ^ Replay with `k` random states from same episode.
              | Future  -- ^ Replay with `k` random states from same episode, that were observed after
              deriving (Eq, Show, Read, Generic, FromJSON, ToJSON)

-- | Run Mode
data Mode = Train   -- ^ Start Agent Training
          | Cont    -- ^ Continue Agent Training
          | Eval    -- ^ Evaluate Agent
          deriving (Eq, Show, Read)

-- | Command Line Arguments
data Args = Args { cktHost    :: String -- ^ Circus Server Host Address
                 , cktPort    :: String -- ^ Circus Server Port
                 , cktId      :: String -- ^ ACE Circuit ID
                 , cktBackend :: String -- ^ ACE Backend / PDK
                 , cktSpace   :: String -- ^ Design Space
                 , algorithm  :: String -- ^ RL Algorithm
                 , buffer     :: String -- ^ Replay Buffer
                 , path       :: String -- ^ Checkpoint Base Path
                 , mlfHost    :: String -- ^ MLFlow Server Host Address
                 , mlfPort    :: String -- ^ MLFlow Server Port
                 , mode       :: String -- ^ Run Mode
                 , config     :: String -- ^ File Path to config.yaml
                 } deriving (Show)

-- | Type Alias for Transition Tuple (state, action, reward, state', done)
type Transition = (T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor)

------------------------------------------------------------------------------
-- Evaluation
------------------------------------------------------------------------------

-- | Calculate success rate given 1D Boolean done Tensor
successRate :: T.Tensor -> Float
successRate dones = rate
  where
    num  = realToFrac . head . T.shape             $ dones
    suc  = realToFrac . head . T.shape . T.nonzero $ dones
    rate = (suc / num) * 100.0

------------------------------------------------------------------------------
-- Convenience / Syntactic Sugar
------------------------------------------------------------------------------

-- | Range from 0 to n - 1
range :: Int -> [Int]
range n = [0 .. (n - 1)]

-- | First of triple
fst' :: (a,b,c) -> a
fst' (a,_,_) = a

-- | Delete multiple elements from a Set
delete' :: (Ord a) => [a] -> S.Set a -> S.Set a
delete' elems set = foldl (flip S.delete) set elems

-- | Helper function creating split indices
splits :: [Int] -> [[Int]]
splits (x:y:zs) = [x .. y] : splits (succ y:zs)
splits [_]      = []
splits []       = []

-- | First of Triple
fst3 :: (a,b,c) -> a
fst3 (a,_,_) = a

-- | Swaps elements of tuple
swap :: (a,b) -> (b,a)
swap (x,y) = (y,x)

-- | Apply a function to both elements of a Tuple
both :: (a -> b) -> (a,a) -> (b,b)
both f (x, y) = (f x, f y)

-- | Uncurry Triple
uncurry3 :: (a -> b -> c -> d) -> (a, b, c) -> d
uncurry3 f (a, b, c) = f a b c

-- | Like `Data.Map.lookup` but for a list of keys.
lookup' :: (Ord k) => [k] -> M.Map k a -> Maybe [a]
lookup' ks m = mapM (`M.lookup` m) ks

-- | Map an appropriate function over a transition tuple
tmap :: (T.Tensor -> T.Tensor) -> Transition -> Transition
tmap f (s, a, r, n, d) = (f s, f a, f r, f n, f d)

-- | Infix div
(//) :: Integral a => a -> a -> a
(//) = div

-- | Infix mod
(%) :: Integral a => a -> a -> a
(%) = mod

------------------------------------------------------------------------------
-- File System
------------------------------------------------------------------------------

-- | Current Timestamp as formatted string
currentTimeStamp :: String -> IO String
currentTimeStamp format = formatTime defaultTimeLocale format <$> getCurrentTime

-- | Current Timestamp with default formatting: "%Y%m%d-%H%M%S"
currentTimeStamp' :: IO String
currentTimeStamp' = currentTimeStamp "%Y%m%d-%H%M%S"

-- | Create a model archive directory for the given algorithm
createModelArchiveDir :: String -> IO String
createModelArchiveDir algorithm = do
    path <- (path' ++) <$> currentTimeStamp'
    createDirectoryIfMissing True path
    pure path
  where
    path' = "./models/" ++ algorithm ++ "/"

-- | Create a model archive directory for the given algorithm, ace id and backend
createModelArchiveDir' :: String -> String -> String -> String -> String 
                       -> String -> IO String
createModelArchiveDir' base alg ace pdk var spc = do
    path <- (path' ++) <$> currentTimeStamp'
    createDirectoryIfMissing True path
    pure path
  where
    path' = base ++ "/" ++ alg ++ "/" ++ ace ++ "-" ++ pdk ++ "-" ++ spc 
                 ++ "-v" ++ var ++ "-"

-- | Optimizer moments at given prefix
saveOptim :: T.Adam -> FilePath -> IO ()
saveOptim optim prefix = do
    T.save (T.m1 optim) (prefix ++ "M1.pt")
    T.save (T.m2 optim) (prefix ++ "M2.pt")

-- | Load Optimizer State
loadOptim :: Int -> Float -> Float -> FilePath -> IO T.Adam
loadOptim iter ??1 ??2 prefix = do
    m1' <- T.load (prefix ++ "M1.pt")
    m2' <- T.load (prefix ++ "M2.pt")
    pure $ T.Adam ??1 ??2 m1' m2' iter

------------------------------------------------------------------------------
-- Neural Networks
------------------------------------------------------------------------------

-- | Calculate weight Limits based on Layer Dimensions
weightLimit :: T.Linear -> Float
weightLimit layer = fanIn ** (- 0.5)
  where
    fanIn = realToFrac . head . T.shape . T.toDependent . NN.weight $ layer

-- | Type of weight initialization
data Initializer = Normal           -- ^ Normally distributed weights
                 | Uniform          -- ^ Uniformally distributed weights
                 | XavierNormal     -- ^ Using T.xavierNormal
                 | XavierUniform    -- ^ Using T.xavierUniform
                 | KaimingNormal    -- ^ Using T.kaimingNormal
                 | KaimingUniform   -- ^ Using T.kaimingUniform
                 | Dirac
                 | Eye
                 | Ones
                 | Zeros
                 | Constant

-- | Weights for a layer given limits and dimensions.
initWeights :: Initializer -> Float -> Float -> [Int] -> IO T.IndependentTensor
initWeights Uniform lo   hi  dims = T.uniform' dims lo hi >>= T.makeIndependent
initWeights Normal  mean std dims = T.normalIO      m' s' >>= T.makeIndependent
  where
    m' = T.toFloat $ T.full' dims mean
    s' = T.toFloat $ T.full' dims std
initWeights XavierNormal   gain _ dims = T.xavierNormal  gain dims 
                                            >>= T.makeIndependent
initWeights XavierUniform  gain _ dims = T.xavierUniform gain dims 
                                            >>= T.makeIndependent
initWeights KaimingNormal  _    _ dims = T.kaimingNormal T.FanIn T.Relu dims 
                                            >>= T.makeIndependent
initWeights KaimingUniform _    _ dims = T.kaimingUniform T.FanIn T.Relu dims 
                                            >>= T.makeIndependent
initWeights _ _ _ _                    = error "Not Implemented"

-- | Initialize Weights of Linear Layer
weightInit :: Initializer -> Float -> Float -> T.Linear -> IO T.Linear
weightInit initType p1 p2 layer = do
    weight' <- initWeights initType p1 p2 dims
    pure T.Linear { NN.weight = weight', NN.bias = bias' }
  where
    dims  = T.shape . T.toDependent . NN.weight $ layer
    bias' = NN.bias layer

-- | Initialize Weights and Bias of Linear Layer
weightInit' :: Initializer -> Float -> Float -> T.Linear -> IO T.Linear
weightInit' initType p1 p2 layer = do
    weight' <- initWeights initType p1 p2 dimsWeights
    bias'   <- initWeights initType p1 p2 dimsBias
    pure T.Linear { NN.weight = weight', NN.bias = bias' }
  where
    dimsWeights = T.shape . T.toDependent . NN.weight $ layer
    dimsBias    = T.shape . T.toDependent . NN.bias   $ layer

-- | Initialize weights uniformally given upper and lower bounds
weightInitUniform :: Float -> Float -> T.Linear -> IO T.Linear
weightInitUniform = weightInit' Uniform

-- | Initialize weights uniformally based on Fan In
weightInitUniform' :: T.Linear -> IO T.Linear
weightInitUniform' layer = weightInit Uniform (-limit) limit layer
  where
    limit = weightLimit layer

-- | Initialize weights normally given mean and std bounds
weightInitNormal :: Float -> Float -> T.Linear -> IO T.Linear
weightInitNormal = weightInit' Normal

-- | Initialize weights normally based on Fan In
weightInitNormal' :: T.Linear -> IO T.Linear
weightInitNormal' layer = weightInit Normal 0.0 limit layer
  where
    limit = weightLimit layer

-- | Softly update parameters from Online Net to Target Net
softUpdate :: T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor
softUpdate ?? t o = (t * (o' - ??)) + (o * ??)
  where
    o' = T.onesLike ??

-- | Softly copy parameters from Online Net to Target Net
softSync :: NN.Parameterized f => T.Tensor -> f -> f -> IO f
softSync ?? target online =  NN.replaceParameters target 
                        <$> mapM T.makeIndependent tUpdate 
  where
    tParams = fmap T.toDependent . NN.flattenParameters $ target
    oParams = fmap T.toDependent . NN.flattenParameters $ online
    tUpdate = zipWith (softUpdate ??) tParams oParams

-- | Hard Copy of Parameter from one net to the other
copySync :: NN.Parameterized f => f -> f -> f
copySync target =  NN.replaceParameters target . NN.flattenParameters
