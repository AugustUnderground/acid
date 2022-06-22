{-# OPTIONS_GHC -Wall #-}

-- | General Replay Buffer Types and TypeClasses
module ALG where

import Torch             (Tensor)
import MLFlow.Extensions (Tracker)
import CKT               (CircusUrl)
import CFG
import Lib

-- | Replay Buffer Interface
class Agent a where
  -- | Save an agent at a given Path
  saveAgent      :: FilePath -> a -> IO a
  -- | Load an agent saved at a Path
  loadAgent      :: Params -> FilePath -> Int -> Int -> Int -> IO a
  -- | Take an action to the best Ability
  act            :: a -> Tensor -> IO Tensor
  -- | Take a noisy action
  act'           :: a -> Tensor -> IO Tensor
  -- | Take an action without any noise
  act''          :: a -> Tensor -> Tensor
  -- | Update Policy
  updatePolicy   :: Params -> CircusUrl -> Tracker -> Int -> [Transition] -> a 
                 -> IO a
