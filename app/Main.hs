{-# OPTIONS_GHC -Wall #-}

module Main where

import Options.Applicative

import Lib
import ACiD

main :: IO ()
main =  execParser opts >>= run
  where
    desc = "Artificial Circuit Designer / Circus Clown"
    opts = info (args <**> helper) 
                (fullDesc <> progDesc desc <> header "ACiD")

-- | Command Line Argument Parser
args :: Parser Args
args = Args <$> strOption ( long "algorithm" 
                         <> short 'l'
                         <> metavar "ALGORITHM" 
                         <> showDefault 
                         <> value "td3"
                         <> help "DRL Algorithm. One of td3 (default), sac, ppo" )
            <*> strOption ( long "buffer" 
                         <> short 'b'
                         <> metavar "BUFFER" 
                         <> showDefault 
                         <> value "HER"
                         <> help "Replay Buffer Type. One of her" )
            <*> strOption ( long "circus-host" 
                         <> short 'H'
                         <> metavar "HOST" 
                         <> showDefault 
                         <> value "localhost"
                         <> help "Circus server host address" )
            <*> strOption ( long "circus-port" 
                         <> short 'P'
                         <> metavar "PORT" 
                         <> showDefault 
                         <> value "6007"
                         <> help "Circus server port" )
            <*> strOption ( long "ace" 
                         <> short 'i'
                         <> metavar "ID" 
                         <> showDefault 
                         <> value "op2"
                         <> help "ACE OP ID. One of op1, op2 (default), op8" )
            <*> strOption ( long "pdk" 
                         <> short 'p'
                         <> metavar "PDK" 
                         <> showDefault 
                         <> value "xh035"
                         <> help "ACE Backend. One of xh018, xh035 (default)" )
            <*> strOption ( long "space"
                         <> short 's'
                         <> metavar "SPACE"
                         <> showDefault
                         <> value "elec"
                         <> help "Design / Action space. elec (default) or geom" )
            <*> strOption ( long "var"
                         <> short 'v'
                         <> metavar "VARIANT"
                         <> showDefault
                         <> value "0"
                         <> help "Circus Environment Variant. 0 = goal (default), 1 = non-goal" )
            <*> strOption ( long "path"
                         <> short 'f'
                         <> metavar "FILE"
                         <> showDefault
                         <> value "./models"
                         <> help "Base Path for Model Checkpoint. Default is ./models" )
            <*> strOption ( long "tracking-host"
                         <> short 'T'
                         <> metavar "HOST"
                         <> showDefault
                         <> value "localhost"
                         <> help "MLFlow tracking server host address" )
            <*> strOption ( long "tracking-port"
                         <> short 'R'
                         <> metavar "PORT"
                         <> showDefault
                         <> value "6008"
                         <> help "MLFlow tracking server port" )
            <*> strOption ( long "mode"
                         <> short 'm'
                         <> metavar "MODE"
                         <> showDefault
                         <> value "Train"
                         <> help "Run Mode. One of Train (default), Continue, Evaluate" )
