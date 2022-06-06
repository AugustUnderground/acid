{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleInstances #-}

-- | Hindsight Experience Replay
module RPB.HER ( Strategy (..)
               , Buffer (..) 
               , empty
               , envSplit
               , epsSplit
               , sampleGoals
               ) where

import qualified RPB
import           Lib
import           CKT                            hiding (url)
import           MLFlow.Extensions
import           ALG
import           ALG.HyperParameters            hiding (d)

import           Control.Monad
import           Control.Applicative            hiding (empty)
import           Prelude                        hiding (drop)
import qualified Torch                     as T
import qualified Torch.Extensions          as T

------------------------------------------------------------------------------
-- Hindsight Experience Replay
------------------------------------------------------------------------------

-- | Hindsight Experience Replay Strategies for choosing Goals
data Strategy = Final   -- ^ Only Final States are additional targets
              | Random  -- ^ Replay with `k` random states encountered so far (basically vanilla)
              | Episode -- ^ Replay with `k` random states from same episode.
              | Future  -- ^ Replay with `k` random states from same episode, that were observed after
  deriving (Show, Eq)

-- | Strict Simple/Naive Replay Buffer
data Buffer a = Buffer { states  :: !a   -- ^ States
                       , actions :: !a   -- ^ Actions
                       , rewards :: !a   -- ^ Rewards
                       , states' :: !a   -- ^ Next States
                       , dones   :: !a   -- ^ Terminal Mask
                       , goals   :: !a   -- ^ Desired Goal
                       , goals'  :: !a   -- ^ Acheived Goal
                       } deriving (Show, Eq)

-- | Hindsight Experience Replay Buffer is a implements `Functor`
instance Functor Buffer where
  fmap f (Buffer s a r s' d g g') = 
         Buffer (f s) (f a) (f r) (f s') (f d) (f g) (f g')

-- | This is badly defined and only so it can use `liftA2`.
instance Applicative Buffer where
  pure a = Buffer a a a a a a a
  (Buffer fs fa fr fs' fd fg fg') <*> (Buffer s a r s' d g g') 
      = Buffer (fs s) (fa a) (fr r) (fs' s') (fd d) (fg g) (fg' g')

-- | Hindsight Experience Replay Buffer implements `ReplayBuffer`
instance RPB.ReplayBuffer (Buffer T.Tensor) where
  -- | See documentation for `size`
  size              = size
  -- | See documentation for `push`
  push              = push
  -- | See documentation for `lookUp'`
  lookUp            = lookUp
  -- | See documentation for `sampleIO`
  sampleIO          = sampleIO
  -- | See documentation for `asTuple`
  asTuple           = asTuple
  -- | See documentation for `collectExperience`
  collectExperience = collectExperience

-- | Construct an empty HER Buffer
empty :: Buffer T.Tensor
empty = Buffer ft ft ft ft bt ft ft
  where
    fopts = T.withDType T.dataType . T.withDevice T.cpu $ T.defaultOpts
    bopts = T.withDType T.Float    . T.withDevice T.cpu $ T.defaultOpts
    ft    = T.asTensor' ([] :: [Float]) fopts
    bt    = T.asTensor' ([] :: [Bool])  bopts

-- | How many Trajectories are currently stored in memory
size :: Buffer T.Tensor -> Int
size = head . T.shape . states

-- | Drop buffer entries exceeding the capacity
drop :: Int -> Buffer T.Tensor -> Buffer T.Tensor
drop cap buf | len < cap = buf
             | otherwise = fmap (T.indexSelect 0 idx) buf
  where
    opts  = T.withDType T.Int32 . T.withDevice T.cpu $ T.defaultOpts
    len   = size buf
    idx   = T.arange (len - cap) len 1 opts

-- | Push one buffer into another one
push :: Int -> Buffer T.Tensor -> Buffer T.Tensor -> Buffer T.Tensor
push cap buf buf' = drop cap $ liftA2 cat buf buf'
  where
    cat a b = T.cat (T.Dim 0) [a,b]
    
-- | Get the given indices from Buffer
lookUp :: [Int] -> Buffer T.Tensor -> Buffer T.Tensor
lookUp idx = fmap (T.indexSelect 0 idx')
  where
    opts = T.withDType T.Int32 . T.withDevice T.cpu $ T.defaultOpts
    idx' = T.asTensor' (idx :: [Int]) opts

-- | Take n random samples from HER Buffer
sampleIO :: Int -> Buffer T.Tensor -> IO (Buffer T.Tensor)
sampleIO num buf = do
    idx <- (T.asValue <$> T.multinomialIO (T.ones' [len]) num False) :: IO [Int]
    pure $ lookUp idx buf
  where
    len = size buf

-- | Split buffer collected from pool by env
envSplit :: Int -> Buffer T.Tensor -> [Buffer T.Tensor]
envSplit ne buf = map (`lookUp` buf) idx
  where
    opts = T.withDType T.Int64 . T.withDevice T.cpu $ T.defaultOpts
    len  = size buf
    idx' = T.reshape [-1,1] (T.arange 0 ne 1  opts) + T.arange 0 len ne opts
    idx  = map (takeWhile (<len)) $ T.asValue idx' :: [[Int]]

-- | Split a buffer into episodes, dropping the last unfinished
epsSplit :: Buffer T.Tensor -> [Buffer T.Tensor]
epsSplit buf@Buffer{..} | T.any dones = map (`lookUp` buf) dix
                        | otherwise   = []
  where
    dones' = T.reshape [-1] . T.squeezeAll . T.nonzero . T.squeezeAll $ dones
    dix    = splits (0 : (T.asValue dones' :: [Int]))

-- | Return (State, Action, Reward, Next State, Done) Tuple
asTuple :: Buffer T.Tensor -> MiniBatch
asTuple Buffer{..} = (s,a,r,n,d)
  where
    s = T.cat (T.Dim 1) [states, goals]
    a = actions
    r = rewards
    n = T.cat (T.Dim 1) [states', goals]
    d = dones

-- | Sample Additional Goals according to Strategy (drop first). `Random` is
-- basically the same as `Episode` you just have to give it the entire buffer,
-- not just the episode.
sampleGoals :: CircusUrl -> Strategy -> Int -> Buffer T.Tensor 
            -> IO (Buffer T.Tensor)
sampleGoals url Final _ buf@Buffer{..} = do
    rewards'   <- calculateReward url states' goal
    let dones' = rewards' + 1.0
        buf'   = Buffer states actions states' rewards' dones' goal goals'
    pure $ push cap buf buf'
  where
    bs         = size buf
    idx        = T.toIntTensor' . (:[]) . pred $ bs
    goal       = T.repeat [bs, 1] $ T.indexSelect 0 idx goals'
    cap        = 2 * bs
sampleGoals url Episode k' buf = do
    idx        <-  map (T.asValue . T.squeezeAll) . T.split 1 (T.Dim 0) 
               <$> T.multinomialIO (T.ones' [bs,bs]) k' False :: IO [[Int]]
    let goal   = T.cat (T.Dim 0) $ map (goals' . (`lookUp` buf)) idx
        rep    = T.full [bs] k' opts
        rbuf   = fmap (T.repeatInterleave' 0 rep) buf
    rewards'   <- calculateReward url (states' rbuf) goal
    let dones' = rewards' + 1.0
        buf'   = Buffer (states rbuf) (actions rbuf) rewards' (states' rbuf) 
                        dones' goal (goals' rbuf)
    pure $ push cap buf buf'
  where
    bs         = size buf
    cap        = bs + (k' * bs)
    opts       = T.withDType T.Int32 . T.withDevice T.cpu $ T.defaultOpts
sampleGoals url Random k' buf = sampleGoals url Episode k' buf
sampleGoals url Future k' buf | k' >= bs   = pure buf
                             | otherwise = do 
    finBuf     <- sampleGoals url Final k' $ lookUp finIdx buf
    idx        <- sequence [  T.asValue . (T.asTensor (bs' :: Int) +) 
                          <$> T.multinomialIO (T.ones' [bs - bs']) k' False 
                           |  bs' <- [ 0 .. bs ], bs' < (bs - k') ]
    let goal   = T.cat (T.Dim 0) $ map (goals' . (`lookUp` buf)) idx
        rep    = T.full [bs - k'] k' opts
        rbuf   = T.repeatInterleave' 0 rep <$> lookUp idx' buf
    rewards'   <- calculateReward url (states' rbuf) goal
    let dones' = rewards' + 1.0
        buf'   = Buffer (states rbuf) (actions rbuf) rewards' (states' rbuf) 
                        dones' goal (goals' rbuf)
        cap    = size buf + size buf' + size finBuf
    pure       $ foldl (push cap) buf [buf', finBuf]
  where
    bs         = size buf
    finIdx     = [ bs - k' .. bs - 1 ]
    idx'       = [ 0 .. bs - k' - 1 ]
    opts       = T.withDType T.Int32 . T.withDevice T.cpu $ T.defaultOpts

-- | Evaluate Policy for T steps and return experience Buffer
collectStep :: (Agent a) => CircusUrl -> Tracker -> Int -> Int -> a -> T.Tensor 
            -> Buffer T.Tensor -> IO (Buffer T.Tensor)
collectStep url _       _    0 _     _ buf = sampleGoals url Future k buf
collectStep url tracker iter t agent s buf = do
    p <- (iter *) <$> numEnvs url
    a <- if p < warmupPeriode
            then randomAction url
            else act' agent s >>= T.detach
    
    (s', g', g, r, d) <- step url a

    trackReward   tracker (iter' !! t') r
    trackEnvState tracker url (iter' !! t')

    let buf' = push bufferSize buf (Buffer s a r s' d g g')

    s_ <- if T.any d then fst3 <$> reset' url d else pure s'

    when (verbose && iter `mod` 10 == 0) do
        putStrLn $ "\tAverage Reward: \t" ++ show (T.mean r)

    collectStep url tracker iter t' agent s_ buf'
  where
    iter' = [(iter * horizonT) .. (iter * 2 * horizonT + 1)]
    t'     = t - 1

-- | Collect experience for a given number of steps
collectExperience :: (Agent a) => CircusUrl -> Tracker -> Int -> a 
                   -> IO (Buffer T.Tensor)
collectExperience url tracker iter agent = do
    s <- fst3 <$> reset url
    collectStep url tracker iter horizonT agent s buffer
  where
    buffer = empty
