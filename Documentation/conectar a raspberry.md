# Conectar a Raspberry

## Ethernet

Conectar cable ethernet al router. Entrar a la configuración del router para saber qué IP tiene la raspberry.

## Wi-Fi hotspot

La raspberry está configurada para crear una red wifi cuando se enciende. Nos podemos conectar así si no podemos usar cable.

user: `catchit`

contra: `catchit1234`

si windows pide un PIN, darle a la opción de usar contraseña, no un pin.

la ip de la raspberry en esta red es: 10.42.0.1


## SSH

abrir una terminal de windows y escribir:

```shell
ssh catchit@10.42.0.1
```

importante poner la ip correcta si usamos ethernet, no será esa.

y accederemos a la consola de comandos de la raspberry

## Actualizar codigo

para descargar los ultimos cambios del repositorio de github:
1. ir a la carpeta de la repo `cd ~/Catch-It`
2. `git pull`
3. nos pedirá un nombre de usuario y la contraseña. el nombre de usuario es el vuestro de github y la contraseña es un token que teneis que crear aquí https://github.com/settings/tokens