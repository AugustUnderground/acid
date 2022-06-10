# ACiD

Artificial Circuit Designer, a 
[circus](https://github.com/augustunderground/circus) clown.

## Dependencies

- [HaskTorch](https://github.com/hasktorch/hasktorch)
- [MLFlow](https://github.com/AugustUnderground/mlflow-hs)
- [Circus](https://github.com/AugustUnderground/circus)

## Command Line Interface

Run a training on `op2` in `xh035` with a `TD3` agent and `HER` buffer in the
`Electric` design space:

```bash
$ stack exec -- acid-exe -l TD3 -b HER -i op2 -p xh035 -v 0 -s elec
```

Here is the short help for the CLI and available options:

```bash
ACiD

Usage: acid-exe [-l|--algorithm ALGORITHM] [-b|--buffer BUFFER]
                [-H|--circus-host HOST] [-P|--circus-port PORT] [-i|--ace ID]
                [-p|--pdk PDK] [-s|--space SPACE] [-v|--var VARIANT]
                [-f|--path FILE] [-T|--tracking-host HOST]
                [-R|--tracking-port PORT] [-m|--mode MODE]
  Artificial Circuit Designer / Circus Clown

Available options:
  -l,--algorithm ALGORITHM DRL Algorithm. One of td3 (default), sac, ppo
                           (default: "td3")
  -b,--buffer BUFFER       Replay Buffer Type. One of her (default: "HER")
  -H,--circus-host HOST    Circus server host address (default: "localhost")
  -P,--circus-port PORT    Circus server port (default: "6007")
  -i,--ace ID              ACE OP ID. One of op1, op2 (default), op8
                           (default: "op2")
  -p,--pdk PDK             ACE Backend. One of xh018, xh035 (default)
                           (default: "xh035")
  -s,--space SPACE         Design / Action space. elec (default) or geom
                           (default: "elec")
  -v,--var VARIANT         Circus Environment Variant. 0 = goal (default), 1 =
                           non-goal (default: "0")
  -f,--path FILE           Base Path for Model Checkpoint. Default is ./models
                           (default: "./models")
  -T,--tracking-host HOST  MLFlow tracking server host address
                           (default: "localhost")
  -R,--tracking-port PORT  MLFlow tracking server port (default: "6008")
  -m,--mode MODE           Run Mode. One of Train (default), Continue, Evaluate
                           (default: "Train")
  -h,--help                Show this help text
```

## TODO

- [X] TD3
- [ ] SAC
- [ ] PPO
- [X] HER
- [ ] PER
- [ ] ERE
- [ ] Hyper Parameters as YAML
