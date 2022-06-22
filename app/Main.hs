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
