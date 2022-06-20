{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- | Utility and Helper functions for EDELWACE
module Lib where

import qualified Data.Map           as M
import qualified Data.Set           as S
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

-- | Run Mode
data Mode = Train       -- ^ Start Agent Training
          | Continue    -- ^ Continue Agent Training
          | Evaluate    -- ^ Evaluate Agent
          deriving (Eq, Show, Read)

-- | Command Line Arguments
data Args = Args { algorithm :: String -- ^ See ALG.Algorithm
                 , memory    :: String -- ^ See RPB.ReplayMemory
                 , cktHost   :: String -- ^ Circus Server Host Address
                 , cktPort   :: String -- ^ Circus Server Port
                 , ace       :: String -- ^ ACE ID
                 , pdk       :: String -- ^ ACE PDK
                 , space     :: String -- ^ Design / Action Space
                 , var       :: String -- ^ (Non-)Goal Environment
                 , cpPath    :: String -- ^ Checkpoint Base Path
                 , mlfHost   :: String -- ^ MLFlow Server Host Address
                 , mlfPort   :: String -- ^ MLFlow Server Port
                 , mode      :: String -- ^ Run Mode
                 } deriving (Show)

-- | Type Alias for Transition Tuple (state, action, reward, state', done)
type Transition = (T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor)

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
delete' elems set = S.unions $ map (`S.delete` set) elems

-- | Helper function creating split indices
splits :: [Int] -> [[Int]]
splits (x:y:zs) = [x .. y] : splits (succ y:zs)
splits [_]      = []
splits []       = []

-- | First of Triple
fst3 :: (a,b,c) -> a
fst3 (a,_,_) = a

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
loadOptim iter β1 β2 prefix = do
    m1' <- T.load (prefix ++ "M1.pt")
    m2' <- T.load (prefix ++ "M2.pt")
    pure $ T.Adam β1 β2 m1' m2' iter

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
softUpdate τ t o = (t * (o' - τ)) + (o * τ)
  where
    o' = T.onesLike τ

-- | Softly copy parameters from Online Net to Target Net
softSync :: NN.Parameterized f => T.Tensor -> f -> f -> IO f
softSync τ target online =  NN.replaceParameters target 
                        <$> mapM T.makeIndependent tUpdate 
  where
    tParams = fmap T.toDependent . NN.flattenParameters $ target
    oParams = fmap T.toDependent . NN.flattenParameters $ online
    tUpdate = zipWith (softUpdate τ) tParams oParams

-- | Hard Copy of Parameter from one net to the other
copySync :: NN.Parameterized f => f -> f -> f
copySync target =  NN.replaceParameters target . NN.flattenParameters
