{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE BlockArguments    #-}
{-# LANGUAGE RecordWildCards   #-}
{-# LANGUAGE FlexibleInstances #-}

-- | General Replay Buffer Types and TypeClasses
module RPB where

import           Prelude             hiding (drop)
import           Control.Applicative hiding (empty)
import           Control.Monad
import           MLFlow.Extensions
import           Lib
import           CKT                 hiding (url)
import           CFG                 hiding (d)
import           ALG
import qualified Torch               as T
import qualified Torch.Extensions    as T

-- | Replay Buffer Interface
class (Functor b) => ReplayBuffer b where
  -- | Return size of current buffer
  size     :: b T.Tensor -> Int
  -- | Push one buffer into another
  push     :: Int -> b T.Tensor -> b T.Tensor -> b T.Tensor
  -- | Look Up given list if indices
  lookUp   :: [Int] -> b T.Tensor -> b T.Tensor
  -- | Take n Random Samples
  sampleIO :: Int -> b T.Tensor -> IO (b T.Tensor)
  -- | Return the Tuple: (s, a, r, s', d) for training
  asTuple  :: b T.Tensor -> (T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor)
  -- | Collect Experiences in Buffer
  collectExperience :: (Agent a) => Params -> CircusUrl 
                    -> Tracker -> Int -> a -> IO (b T.Tensor)

-- | Generate a list of uniformly sampled minibatches
randomBatches :: (ReplayBuffer b) => Int -> Int -> b T.Tensor -> IO [Transition]
randomBatches nb bs buf = do
    idx <-  (map T.asValue . T.split bs (T.Dim 0) 
                    <$> T.multinomialIO (T.ones' [bl]) num rpl
            ) :: IO [[Int]]

    pure $ map (asTuple . (`lookUp` buf)) idx
  where
    bl  = size buf
    num = nb * bs
    rpl = num > bl

-- | Vanilla Replay Buffer
data Buffer a = Buffer { states  :: !a   -- ^ States
                       , actions :: !a   -- ^ Actions
                       , rewards :: !a   -- ^ Rewards
                       , states' :: !a   -- ^ Next States
                       , dones   :: !a   -- ^ Terminal Mask
                       } deriving (Show, Eq)

-- | Vanilla ReplayBuffer implements `functor`
instance Functor Buffer where
  fmap f (Buffer s a r s' d) = Buffer (f s) (f a) (f r) (f s') (f d)

-- | This is badly defined and only so it can use `liftA2`.
instance Applicative Buffer where
  pure a = Buffer a a a a a
  (Buffer fs fa fr fs' fd) <*> (Buffer s a r s' d) 
      = Buffer (fs s) (fa a) (fr r) (fs' s') (fd d)

-- | Vanilla Replay Buffer implements `ReplayBuffer`
instance ReplayBuffer Buffer where
  -- | See documentation for `size'`
  size              = size'
  -- | See documentation for `push'`
  push              = push'
  -- | See documentation for `lookUp'`
  lookUp            = lookUp'
  -- | See documentation for `sampleIO'`
  sampleIO          = sampleIO'
  -- | See documentation for `asTuple'`
  asTuple           = asTuple'
  -- | See documentation for `collectExperience'`
  collectExperience = collectExperience'

-- | Create a new, empty Buffer on the CPU
empty :: Buffer T.Tensor
empty = Buffer ft ft ft ft bt
  where
    opts = T.withDType T.Float . T.withDevice T.cpu $ T.defaultOpts
    ft   = T.asTensor' ([] :: [Float]) opts
    bt   = T.asTensor' ([] :: [Bool])  opts

-- | How many Trajectories are currently stored in memory
size' :: Buffer T.Tensor -> Int
size' = head . T.shape . states

-- | Drop number of entries from the beginning of the Buffer
drop :: Int -> Buffer T.Tensor -> Buffer T.Tensor
drop cap buf = fmap (T.indexSelect 0 idx) buf
  where
    opts  = T.withDType T.Int32 . T.withDevice T.cpu $ T.defaultOpts
    len   = size buf
    idx   = if len < cap
               then T.arange      0      len 1 opts
               else T.arange (len - cap) len 1 opts

-- | Push one buffer into another one
push' :: Int -> Buffer T.Tensor -> Buffer T.Tensor -> Buffer T.Tensor
push' cap buf buf' = drop cap $ liftA2 cat buf buf'
  where
    cat a b = T.cat (T.Dim 0) [a,b]

-- | Get the given indices from Buffer
lookUp' :: [Int] -> Buffer T.Tensor -> Buffer T.Tensor
lookUp' idx = fmap (T.indexSelect 0 idx')
  where
    opts = T.withDType T.Int32 . T.withDevice T.cpu $ T.defaultOpts
    idx' = T.asTensor' (idx :: [Int]) opts

-- | Take n random samples from Buffer
sampleIO' :: Int -> Buffer T.Tensor -> IO (Buffer T.Tensor)
sampleIO' num buf = do
    idx <- (T.asValue <$> T.multinomialIO (T.ones' [len]) num False) :: IO [Int]
    pure $ lookUp' idx buf
  where
    len = size buf

-- | Return (State, Action, Reward, Next State, Done) Tuple
asTuple' :: Buffer T.Tensor -> Transition
asTuple' (Buffer s a r n d) = (s,a,r,n,d)

-- | Evaluate Policy for T steps and return experience Buffer
collectStep :: (Agent a) => Params -> CircusUrl -> Tracker -> Int -> Int -> a 
            -> T.Tensor -> Buffer T.Tensor -> IO (Buffer T.Tensor)
collectStep _            _   _       _    0 _     _ buf = pure buf
collectStep p@Params{..} url tracker iter t agent s buf = do
    p' <- (iter *) <$> numEnvs url
    a <- if p' % explFreq == 0
            then randomAction url
            else act' agent s >>= T.detach
    
    (s', _, _, r, d) <- step url a

    trackReward   tracker (iter' !! t') r
    trackEnvState tracker url (iter' !! t')

    let buf' = push bufferSize buf (Buffer s a r s' d)
    
    s_ <- if T.any d then fst3 <$> reset' url d else pure s'

    when (verbose && iter % 10 == 0) do
        putStrLn $ "\tAverage Reward: \t" ++ show (T.mean r)

    collectStep p url tracker iter t' agent s_ buf'
  where
    iter' = [(iter * horizonT) .. (iter * 2 * horizonT + 1)]
    t'     = t - 1

-- | Collect experience for a given number of steps
collectExperience' :: (Agent a) => Params -> CircusUrl -> Tracker -> Int -> a 
                   -> IO (Buffer T.Tensor)
collectExperience' p@Params{..} url tracker iter agent = do
    (obs, _, _) <- reset url
    collectStep p url tracker iter horizonT agent obs buf
  where
    buf = empty
