# Docker configuration
To run application with Docker you need it installed and configured, please see [this](https://docs.docker.com/engine/install/) for a reference.

## Persistent volumes
If you want to have persistent volumes you can:

* use `docker-compose.dev.yml` file for Docker Compose, e.g. `docker-compose -f docker-compose.dev.yml run rpasdt gui`
* copy `docker-compose.dev.yml` and name it `docker-compose.override.yml`. It is gitignored so you won't add this to repo by accident.

## Run application in GUI mode
To run application in GUI mode you need to enable a $DISPLAY port forwarding. Based on the system you use it should be done differently.

### Windows
You need to install and configure **VcXsrv** server, please use [this](https://dev.to/darksmile92/run-gui-app-in-linux-docker-container-on-windows-host-4kde) for the reference.
Then please update `docker-compose.dev.yml` configuration with the correct value for **$DISPLAY**.
```
 docker:
    environment:
      - DISPLAY={YOUR_IP}:0.0
```  
### Linux
It is set correctly set in `docker-compose.dev.yml`.
``` docker:
    environment:
      - DISPLAY=$DISPLAY
```
## Build images
Use provided Docker Compose files to build `rpasdt` images.
To do so, run the following command in `docker` directory:

```shell
docker-compose build
```
## Commands
### Run application with GUI
```
docker-compose run rpasdt gui
```
or just by
```
docker-compose up
```
### Print help
```
docker-compose run rpasdt help
```
### Run CLI command
```
docker-compose run rpasdt cli <command>
```
For CLI commands list please refer [here](cli.md).
### Run Python shell
```
docker-compose run rpasdt python
```

### Open an interactive shell in the container
```
docker-compose run rpasdt shell
```
### Generate a package
```
docker-compose run rpasdt package
```
### Generate docs
```
docker-compose run rpasdt makedoc
```
## References

* [Docker-Compose Docs](https://docs.docker.com/compose/)
* [Docker Docs](https://docs.docker.com/)
