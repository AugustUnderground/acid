{-# OPTIONS_GHC -Wall #-}

-- | Twin Delayed Deep Deterministic Policy Gradient Algorithm Defaults
module ALG.HyperParameters where

import qualified Torch            as T
import qualified Torch.Extensions as T

------------------------------------------------------------------------------
--  General Default Settings
------------------------------------------------------------------------------

-- | Print verbose debug output
verbose :: Bool
verbose     = True
-- | Number of episodes to play
numEpisodes :: Int
numEpisodes = 200
-- | Maximum Number of Steps per Episode
horizonT :: Int
horizonT    = 50
-- | Number of epochs to train
numEpochs :: Int
numEpochs   = 40
-- | Mini batch size
batchSize :: Int
batchSize   = 256
-- | Random seed for reproducibility
rngSeed :: Int
rngSeed     = 666

------------------------------------------------------------------------------
--  Circus Environment Settings
------------------------------------------------------------------------------
           
-- | Action space lower bound
actionLow :: Float
actionLow  = - 1.0
-- | Action space upper bound
actionHigh :: Float
actionHigh = 1.0

------------------------------------------------------------------------------
--  Algorithm Hyper Parameters
------------------------------------------------------------------------------

-- | Policy and Target Update Delay
d :: Int
d           = 2
-- | Noise clipping
c :: Float
c           = 0.5
-- | Discount Factor
γ :: T.Tensor
γ           = T.toTensor (0.99 :: Float)
-- | Soft Update coefficient (sometimes "polyak") of the target 
-- networks τ ∈ [0,1]
τ :: T.Tensor
τ           = T.toTensor (5.0e-3 :: Float)
-- | Decay Period
decayPeriod :: Int
decayPeriod = 10 ^ (5 :: Int)
-- | Noise Clipping Minimum
σMin :: Float
σMin        = 1.0
-- | Noise Clipping Maximum
σMax :: Float
σMax        = 1.0
-- | Evaluation Noise standard deviation (σ~)
σEval :: T.Tensor
σEval       = T.toTensor ([0.2] :: [Float])
-- | Action Noise standard deviation
σAct :: T.Tensor
σAct        = T.toTensor ([0.1] :: [Float])
-- | Noise Clipping
σClip :: Float
σClip       = 0.5

------------------------------------------------------------------------------
-- Neural Network Parameter Settings
------------------------------------------------------------------------------

-- | Number of units per hidden layer
hidDim :: Int
hidDim        = 256
-- | Initial weights
wInit :: Float
wInit         = 3.0e-4
-- | Actor Learning Rate
ηφ :: T.Tensor
ηφ            = T.toTensor (1.0e-3 :: Float)
-- | Critic Learning Rate
ηθ :: T.Tensor
ηθ            = T.toTensor (1.0e-3 :: Float)
-- | ADAM Hyper Parameter β1
β1   :: Float
β1            = 0.9
-- | ADAM Hyper Parameter β2
β2   :: Float
β2            = 0.99
-- | Leaky ReLU Slope
negativeSlope :: Float
negativeSlope = 0.01

------------------------------------------------------------------------------
--  Memory / Replay Buffer Settings
------------------------------------------------------------------------------

-- | Replay Buffer Size
bufferSize :: Int
bufferSize    = 10 ^ (6 :: Int)
-- | Frequency of random exploration Episodes
rngEpisodeFreq :: Int
rngEpisodeFreq = 10

------------------------------------------------------------------------------
-- Hindsight Experience Replay Settings
------------------------------------------------------------------------------

-- | Number of Additional Targets to sample
k :: Int 
k = 4
