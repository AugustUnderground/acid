{-# OPTIONS_GHC -Wall #-}

-- | General Replay Buffer Types and TypeClasses
module ALG where

import Torch             (Tensor)
import MLFlow.Extensions (Tracker)
import CKT               (CircusUrl)
import Lib

-- | Available Algorithms
data Algorithm = TD3 -- ^ Twin Delayed Deep Deterministic Policy Gradient
               | SAC -- ^ Soft Actor Critic
               | PPO -- ^ Proximal Policy Optimization
               deriving (Eq, Show, Read)

-- | Replay Buffer Interface
class Agent a where
  -- | Save an agent at a given Path
  saveAgent      :: FilePath -> a -> IO a
  -- | Load an agent saved at a Path
  loadAgent      :: FilePath -> Int -> Int -> Int -> IO a
  -- | Take an action to the best Ability
  act            :: a -> Tensor -> IO Tensor
  -- | Take a noisy action
  act'           :: a -> Tensor -> IO Tensor
  -- | Take an action without any noise
  act''          :: a -> Tensor -> Tensor
  -- | Update Policy
  updatePolicy   :: CircusUrl -> Tracker -> Int -> [Transition] -> a -> IO a
