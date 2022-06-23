{-# OPTIONS_GHC -Wall #-}

module Main where

import Options.Applicative

import Lib
import ACiD

main :: IO ()
main = execParser opts >>= run
  where
    desc = "Artificial Circuit Designer / Circus Clown"
    opts = info (args <**> helper) 
                (fullDesc <> progDesc desc <> header "ACiD")

-- | Command Line Argument Parser
args :: Parser Args
args = Args <$> strOption ( long "circus-host" 
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
            <*> strOption ( long "circus-id" 
                         <> short 'i'
                         <> metavar "ID" 
                         <> showDefault 
                         <> value "op2"
                         <> help "Circuit ID" )
            <*> strOption ( long "circus-backend" 
                         <> short 'p'
                         <> metavar "PDK" 
                         <> showDefault 
                         <> value "xh035"
                         <> help "Backend / PDK" )
            <*> strOption ( long "circus-space" 
                         <> short 's'
                         <> metavar "SPACE" 
                         <> showDefault 
                         <> value "elec"
                         <> help "Design Space" )
            <*> strOption ( long "algorithm" 
                         <> short 'l'
                         <> metavar "ALG" 
                         <> showDefault 
                         <> value "TD3"
                         <> help "RL Algorithm" )
            <*> strOption ( long "buffer" 
                         <> short 'b'
                         <> metavar "BUF" 
                         <> showDefault 
                         <> value "HER"
                         <> help "Replay Buffer" )
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
            <*> strOption ( long "cfg"
                         <> short 'c'
                         <> metavar "YAML"
                         <> value "./config/td3.yaml"
                         <> showDefault
                         <> help "YAML Config File" )
