# ACiD

Artificial Circuit Designer, a 
[circus](https://github.com/augustunderground/circus) clown.

## Dependencies

- [HaskTorch](https://github.com/hasktorch/hasktorch)
- [MLFlow](https://github.com/AugustUnderground/mlflow-hs)
- [Circus](https://github.com/AugustUnderground/circus)

## Command Line Interface

**After** starting a [circus](https://github.com/AugustUnderground/circus)
server and the [mlflow](https://github.com/AugustUnderground/mlflow-hs)
tracking server acid can be started with the example configuration 
`./config/td3.yaml`:

```bash
$ stack exec -- acid-exe -c ./config/td3.yaml
```

Here is the short help for the CLI and available options:

```bash
ACiD

Usage: acid-exe [-H|--circus-host HOST] [-P|--circus-port PORT] [-f|--path FILE]
                [-T|--tracking-host HOST] [-R|--tracking-port PORT]
                [-m|--mode MODE] [-c|--cfg YAML]
  Artificial Circuit Designer / Circus Clown

Available options:
  -H,--circus-host HOST    Circus server host address (default: "localhost")
  -P,--circus-port PORT    Circus server port (default: "6007")
  -f,--path FILE           Base Path for Model Checkpoint. Default is ./models
                           (default: "./models")
  -T,--tracking-host HOST  MLFlow tracking server host address
                           (default: "localhost")
  -R,--tracking-port PORT  MLFlow tracking server port (default: "6008")
  -m,--mode MODE           Run Mode. One of Train (default), Continue, Evaluate
                           (default: "Train")
  -c,--cfg YAML            YAML Config File (default: "./config/td3.yaml")
  -h,--help                Show this help text
```

## TODO

- [X] TD3
- [ ] SAC
- [ ] PPO
- [X] HER
- [ ] PER
- [ ] ERE
- [X] Hyper Parameters as YAML
