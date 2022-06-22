{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE TypeApplications    #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- | Extensions to Torch
module Torch.Extensions where

import           GHC.Float                       (float2Double)
import qualified Torch                     as T
import qualified Torch.Lens                as TL
import qualified Torch.Functional.Internal as T  (where', nan_to_num, repeatInterleave)

------------------------------------------------------------------------------
-- Convenience / Syntactic Sugar
------------------------------------------------------------------------------

-- | Swaps the arguments of HaskTorch's foldLoop around
foldLoop' :: Int -> (a -> Int -> IO a) -> a -> IO a
foldLoop' i f m = T.foldLoop m i f

-- | Because snake_case sucks
nanToNum :: Float -> Float -> Float -> T.Tensor -> T.Tensor
nanToNum nan' posinf' neginf' self = T.nan_to_num self nan posinf neginf
  where
    nan    = float2Double nan'
    posinf = float2Double posinf'
    neginf = float2Double neginf'

-- | Default limits for `nanToNum`
nanToNum' :: T.Tensor -> T.Tensor
nanToNum' self = T.nan_to_num self nan posinf neginf
  where
    nan    = 0.0 :: Double
    posinf = float2Double (2.0e32 :: Float)
    neginf = float2Double (-2.0e32 :: Float)

-- | Default limits for `nanToNum` (0.0)
nanToNum'' :: T.Tensor -> T.Tensor
nanToNum'' self = T.nan_to_num self nan posinf neginf
  where
    nan    = 0.0 :: Double
    posinf = 0.0 :: Double
    neginf = 0.0 :: Double

-- | GPU Tensor filled with Float value
fullLike' :: T.Tensor -> Float -> T.Tensor
fullLike' self num = T.onesLike self * num'
  where
    opts = T.withDType T.Int32 . T.withDevice (T.device self) $ T.defaultOpts
    num' = T.asTensor' num opts

-- | Select index with [Int] from GPU tensor
indexSelect'' :: Int -> [Int] -> T.Tensor -> T.Tensor
indexSelect'' dim idx ten = ten'
  where
    opts = T.withDType T.Int32 . T.withDevice (T.device ten) $ T.defaultOpts
    idx' = T.asTensor' idx opts
    ten' = T.indexSelect dim idx' ten

-- | Torch.where' with fixed type for where'
where'' :: T.Tensor -> (T.Tensor -> T.Tensor) -> T.Tensor -> T.Tensor
where'' msk fn self = T.where' msk (fn self) self

-- | Syntactic sugar for HaskTorch's `repeatInterleave` so it can more easily
-- be fmapped.
repeatInterleave' :: Int -> T.Tensor -> T.Tensor -> T.Tensor
repeatInterleave' dim rep self = T.repeatInterleave self rep dim

-- | Helper function creating split indices as gpu int tensor
-- splits' :: [Int] -> [T.Tensor]
-- splits' = map tit . splits
--   where
--     tit i = T.asTensor (i :: Int)

-- | Split Tensor into list of Tensors along dimension
splitDim :: Int -> T.Tensor -> [T.Tensor]
splitDim dim self = T.chunk size (T.Dim dim) self
    where
      size = T.shape self !! dim

-- | Create Boolean Mask Tensor from list of indices.
boolMask :: Int -> [Int] -> T.Tensor
boolMask len idx = mask
  where
    mask = T.toDType T.Bool . T.asTensor $ map (`elem` idx) [0 .. (len - 1)]

-- | Create a Boolean Mask Tensor from index Tensor
boolMask' :: Int -> T.Tensor -> T.Tensor
boolMask' len idx = mask
  where
    idx' = T.squeezeAll idx
    mask = T.anyDim (T.Dim 0) False 
         $ T.eq (T.arange' 0 len 1) (T.reshape [-1,1] idx')

------------------------------------------------------------------------------
-- Data Conversion
------------------------------------------------------------------------------

-- | GPU 1
gpu :: T.Device
gpu = T.Device T.CUDA 1

-- | CPU 0
cpu :: T.Device
cpu = T.Device T.CPU 0

-- | Default Tensor Data Type
dataType :: T.DType
dataType = T.Float

-- | Convert an Array to a Tensor on GPU
toTensor :: T.TensorLike a => a -> T.Tensor
toTensor t = T.asTensor' t opts
  where
    opts = T.withDType dataType . T.withDevice gpu $ T.defaultOpts

-- | Convert an Array to a Tensor on CPU
toTensor' :: T.TensorLike a => a -> T.Tensor
toTensor' t = T.asTensor' t opts
  where
    opts = T.withDType dataType . T.withDevice cpu $ T.defaultOpts

-- | Convert an Array to a Tensor on GPU
toIntTensor :: T.TensorLike a => a -> T.Tensor
toIntTensor t = T.asTensor' t opts
  where
    opts = T.withDType T.Int32 . T.withDevice gpu $ T.defaultOpts

-- | Convert an Array to a Tensor on CPU
toIntTensor' :: T.TensorLike a => a -> T.Tensor
toIntTensor' t = T.asTensor' t opts
  where
    opts = T.withDType T.Int32 . T.withDevice cpu $ T.defaultOpts

-- | Create an empty Float Tensor on GPU
empty :: T.Tensor
empty = T.asTensor' ([] :: [Float]) opts
  where
    opts = T.withDType dataType . T.withDevice gpu $ T.defaultOpts

-- | Create an empty Float Tensor on CPU
empty' :: T.Tensor
empty' = T.asTensor' ([] :: [Float]) opts
  where
    opts = T.withDType dataType . T.withDevice cpu $ T.defaultOpts

-- | Convert a Scalar to a Tensor on GPU
toScalar :: Float -> T.Tensor
toScalar t = T.asTensor' t opts
  where
    opts = T.withDType dataType . T.withDevice gpu $ T.defaultOpts

-- | Convert a Scalar to a Tensor on CPU
toScalar' :: Float -> T.Tensor
toScalar' t = T.asTensor' t opts
  where
    opts = T.withDType dataType . T.withDevice cpu $ T.defaultOpts

-- | Convert model to Double on GPU
toDouble :: forall a. TL.HasTypes a T.Tensor => a -> a
toDouble = TL.over (TL.types @ T.Tensor @a) opts
  where
    opts = T.toDevice gpu . T.toType T.Double

-- | Convert model to Double on CPU
toDouble' :: forall a. TL.HasTypes a T.Tensor => a -> a
toDouble' = TL.over (TL.types @ T.Tensor @a) opts
  where
    opts = T.toDevice cpu . T.toType T.Double

-- | Convert model to Float on CPU
toFloat :: forall a. TL.HasTypes a T.Tensor => a -> a
toFloat = TL.over (TL.types @ T.Tensor @a) opts
  where
    opts = T.toDevice gpu . T.toType T.Float

-- | Convert model to Float on CPU
toFloat' :: forall a. TL.HasTypes a T.Tensor => a -> a
toFloat' = TL.over (TL.types @ T.Tensor @a) opts
  where
    opts = T.toDevice cpu . T.toType T.Float

------------------------------------------------------------------------------
-- Statistics
------------------------------------------------------------------------------

-- | Generate a Tensor of random Integers on GPU
randomInts :: Int -> Int -> Int -> IO T.Tensor
randomInts lo hi num = T.randintIO lo hi [num] opts 
  where
    opts = T.withDType T.Int64 . T.withDevice gpu $ T.defaultOpts

-- | Generate a Tensor of random Integers on CPU
randomInts' :: Int -> Int -> Int -> IO T.Tensor
randomInts' lo hi num = T.randintIO lo hi [num] opts 
  where
    opts = T.withDType T.Int64 . T.withDevice cpu $ T.defaultOpts

-- | Generate Normally Distributed Random values given dimensions
normal' :: [Int] -> IO T.Tensor
normal' dims = T.randnIO dims opts
  where
    opts = T.withDType T.Float . T.withDevice gpu $ T.defaultOpts

-- | Generate Uniformally distributed values in a given range
uniform' :: [Int] -> Float -> Float -> IO T.Tensor
uniform' shape lo hi = unscale <$> T.randIO shape opts
  where
    opts = T.withDevice gpu T.defaultOpts
    xMin = toTensor lo
    xMax = toTensor hi
    unscale x = x * (xMax - xMin) + xMin

-- | Rescale tensor s.t. mean = 0.0 and std = 1.0
rescale :: T.Tensor -> T.Tensor
rescale x = x'
  where
    (σ,μ) = T.stdMeanDim (T.Dim  0) True T.KeepDim x
    x'    = (x - μ) / σ
