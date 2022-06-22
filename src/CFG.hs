{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE DeriveGeneric     #-}
{-# LANGUAGE BlockArguments    #-}
{-# LANGUAGE RecordWildCards   #-}
{-# LANGUAGE OverloadedStrings #-}

-- | YAML Configurations
module CFG where

import           Lib
import           CKT
import qualified CFG.Default           as Default
import           GHC.Generics                     hiding (Meta)
import qualified Data.ByteString.Char8 as BS
import           Data.Yaml
import qualified Torch                 as T

------------------------------------------------------------------------------
-- Reading/Writing, Encoding/Decoding
------------------------------------------------------------------------------

-- | Read a Tensor from float
read' :: Float -> Maybe T.Tensor
read' x = Just (T.asTensor [x])

-- | Decode / Parse ByteString to Meta
parseConfig' :: BS.ByteString -> Config
parseConfig' bs = cfg
    where parsed = decodeEither' bs :: Either ParseException Config
          cfg    = case parsed of (Left  err') -> error (show err')
                                  (Right cfg') ->             cfg'

-- | Decode / Parse Config File
parseConfig :: FilePath -> IO Config
parseConfig path = parseConfig' <$> BS.readFile path

------------------------------------------------------------------------------
-- Data Types
------------------------------------------------------------------------------

-- | YAML Config File
data Config = Config { meta            :: Meta            -- ^ Meta Information
                     , hyperParameters :: HyperParameters -- ^ Hyper Parameters
                     } deriving (Generic, Show)

-- | Config JSON Parse Instance
instance FromJSON Config where
  parseJSON (Object v) = do
      Config <$> v .: "meta"
             <*> v .: "hyper-parameters"
  parseJSON _ = fail "Expected an Object"

-- | JSON Config Parse Instance
instance ToJSON Config where
  toJSON Config{..} = object [ "meta"             .= meta
                             , "hyper-parameters" .= hyperParameters
                             ]

-- | Meta Information
data Meta = Meta { verbose     :: Bool          -- ^ Print verbose debug output
                 , rngSeed     :: Int           -- ^ Random seed for reproducibility
                 , numEpisodes :: Int           -- ^ Number of episodes to play
                 , horizonT    :: Int           -- ^ Maximum Number of Steps per Episode
                 , algorithm   :: Algorithm     -- ^ ACiD Algorithm ID
                 , buffer      :: ReplayMemory  -- ^ ACiD Buffer ID
                 , aceId       :: Circuit       -- ^ ACE Single-Ended OpAmp ID
                 , aceBackend  :: PDK           -- ^ ACE Backend / PDK
                 , space       :: Space         -- ^ Design / Action Space
                 , variant     :: Int           -- ^ Environment Variant
                 } deriving (Generic, Show)

-- | Meta JSON Parse Instance
instance FromJSON Meta where
  parseJSON (Object v) = do Meta <$> v .: "verbose"      .!= Default.verbose     -- True
                                 <*> v .: "rng-seed"     .!= Default.rngSeed     -- 666
                                 <*> v .: "num-episodes" .!= Default.numEpisodes -- 100
                                 <*> v .: "horizon"      .!= Default.horizonT    -- 50
                                 <*> v .: "algorithm"    .!= TD3
                                 <*> v .: "buffer"       .!= HER
                                 <*> v .: "ace-id"       .!= OP2
                                 <*> v .: "ace-backend"  .!= XH035
                                 <*> v .: "space"        .!= Electric
                                 <*> v .: "variant"
  parseJSON _ = fail "Expected an Object"

-- | JSON Meta Parse Instance
instance ToJSON Meta where
  toJSON Meta{..} = object [ "verbose"      .=      verbose     
                           , "rng-seed"     .=      rngSeed     
                           , "num-episodes" .=      numEpisodes 
                           , "horizon"      .=      horizonT    
                           , "algorithm"    .= show algorithm   
                           , "buffer"       .= show buffer      
                           , "ace-id"       .= show aceId       
                           , "ace-backend"  .= show aceBackend  
                           , "space"        .= show space       
                           , "variant"      .=      variant     
                           ]

-- | Hyper Parameters
data HyperParameters = HyperParameters 
        { d          :: Int      -- ^ Policy and Target Update Delay
        , c          :: Float    -- ^ Noise clipping
        , γ          :: T.Tensor -- ^ Discount Factor (Tensor)
        , τ          :: T.Tensor -- ^ Soft Update coefficient ("polyak") τ ∈ [0,1]
        , decay      :: Int      -- ^ Decay Period
        , σMin       :: Float    -- ^ Noise Clipping Minimum
        , σMax       :: Float    -- ^ Noise Clipping Maximum
        , σEval      :: T.Tensor -- ^ Evaluation Noise standard deviation (σ~) (Tensor)
        , σAct       :: T.Tensor -- ^ Action Noise standard deviation (Tensor)
        , σClip      :: Float    -- ^ Noise Clipping
        , hidDim     :: Int      -- ^ Number of units per hidden layer
        , wInit      :: Float    -- ^ Initial weights
        , ηφ         :: T.Tensor -- ^ Actor Learning Rate (Tensor)
        , ηθ         :: T.Tensor -- ^ Critic Learning Rate (Tensor)
        , β1         :: Float    -- ^ ADAM Hyper Parameter β1
        , β2         :: Float    -- ^ ADAM Hyper Parameter β2
        , lreluSlope :: Float    -- ^ Leaky ReLU Slope
        , k          :: Int      -- ^ Number of Additional Targets to sample
        , strategy   :: Strategy -- ^ Goal Sampling Strategy
        , actionLow  :: Float    -- ^ Action space lower bound
        , actionHigh :: Float    -- ^ Action space upper bound
        , numEpochs  :: Int      -- ^ Number of epochs to train
        , explFreq   :: Int      -- ^ Frequency of random exploration episodes
        , evalFreq   :: Int      -- ^ Frequency of random exploration episodes
        , bufferSize :: Int      -- ^ Replay Buffer Size
        , batchSize  :: Int      -- ^ Mini batch size
        } deriving (Generic, Show)

-- | Hyper Parameters JSON Parse Instance
instance FromJSON HyperParameters where
  parseJSON (Object v) = do HyperParameters 
    <$>            v .: "d"           .!= Default.d             -- 2
    <*>            v .: "c"           .!= Default.c             -- 0.5
    <*> (read' <$> v .: "γ")          .!= Default.γ             -- Tensor [] 0.99
    <*> (read' <$> v .: "τ")          .!= Default.τ             -- Tensor [] 5.0e-3
    <*>            v .: "decay"       .!= Default.decayPeriod   -- 10 ^ (5 :: Int)
    <*>            v .: "σ-min"       .!= Default.σMin          -- 1.0
    <*>            v .: "σ-max"       .!= Default.σMax          -- 1.0
    <*> (read' <$> v .: "σ-eval")     .!= Default.σEval         -- 0.2
    <*> (read' <$> v .: "σ-act")      .!= Default.σAct          -- 0.1
    <*>            v .: "σ-clip"      .!= Default.σClip         -- 0.5
    <*>            v .: "hidden-dim"  .!= Default.hidDim        -- 256
    <*>            v .: "w-init"      .!= Default.wInit         -- 3.0e-4
    <*> (read' <$> v .: "ηφ")         .!= Default.ηφ            -- Tensor [] 1.0e-3
    <*> (read' <$> v .: "ηθ")         .!= Default.ηθ            -- Tensor [] 1.0e-3
    <*>            v .: "β1"          .!= Default.β1            -- 0.9
    <*>            v .: "β2"          .!= Default.β2            -- 0.99
    <*>            v .: "lrelu-slope" .!= Default.negativeSlope -- 0.01
    <*>            v .: "k"           .!= Default.k             -- 4
    <*>            v .: "strategy"    .!= Future
    <*>            v .: "action-low"  .!= Default.actionLow     -- (-1.0)
    <*>            v .: "action-high" .!= Default.actionHigh    -- 1.0
    <*>            v .: "num-epochs"  .!= Default.numEpochs     -- 40
    <*>            v .: "expl-freq"   .!= Default.explFreq      -- 50
    <*>            v .: "eval-freq"   .!= Default.evalFreq      -- 50
    <*>            v .: "buffer-size" .!= Default.bufferSize    -- 10 ^ (6 :: Int)
    <*>            v .: "batch-size"  .!= Default.batchSize     -- 256
  parseJSON _ = fail "Expected an Object"

-- | JSON Hyper Parameters Parse Instance
instance ToJSON HyperParameters where
  toJSON HyperParameters{..} = object 
        [ "d"           .=                 d
        , "c"           .=                 c
        , "γ"           .= show (T.asValue γ             :: Float)
        , "τ"           .= show (T.asValue τ             :: Float)
        , "decay"       .=                 decay
        , "σ-min"       .=                 σMin
        , "σ-max"       .=                 σMax
        , "σ-eval"      .= show (T.asValue σEval         :: Float)
        , "σ-act"       .= show (T.asValue σAct          :: Float)
        , "σ-clip"      .=                 σClip
        , "hidden-dim"  .=                 hidDim
        , "w-init"      .=                 wInit
        , "ηφ"          .= show (T.asValue ηφ            :: Float)
        , "ηθ"          .= show (T.asValue ηθ            :: Float)
        , "β1"          .=                 β1
        , "β2"          .=                 β2
        , "lrelu-slope" .=                 lreluSlope
        , "k"           .=                 k
        , "strategy"    .=                 show strategy
        , "action-low"  .=                 actionLow
        , "action-high" .=                 actionHigh
        , "num-epochs"  .=                 numEpochs
        , "expl-freq"   .=                 explFreq
        , "eval-freq"   .=                 evalFreq
        , "buffer-size" .=                 bufferSize
        , "batch-size"  .=                 batchSize
        ]
