# Configuration
To run application with Docker you need it installed and configured, please see [this](https://docs.docker.com/engine/install/) for a reference.

## Windows
To run application in GUI mode you need to enable a $DISPLAY port forwarding to do this you need to install and configure **VcXsrv** server, please use [this](https://dev.to/darksmile92/run-gui-app-in-linux-docker-container-on-windows-host-4kde) for the reference. Then please update `docker-compose.dev.yml` configuration with the correct value for **$DISPLAY**:
```
 environment:
      - DISPLAY={YOUR_IP}:0.0
```  


## Build images
Use provided Docker Compose files to build `rpasdt` images.
To do so, run the following command in `docker` directory:

```shell
docker-compose build
```


# Commands
## Run application with GUI
```
docker-compose run rpasdt gui
```
or just by
```
docker-compose up
```
## Print help
```
docker-compose run rpasdt help
```
## Run CLI command
```
docker-compose run rpasdt cli <command>
```
For CLI commands list please refer [here](cli.md).
## Run Python shell
```
docker-compose run rpasdt python
```

## Open an interactive shell in the container
```
docker-compose run rpasdt shell
```
## Generate a package
```
docker-compose run rpasdt package
```

## References

* [Docker-Compose Docs](https://docs.docker.com/compose/)
* [Docker Docs](https://docs.docker.com/)
