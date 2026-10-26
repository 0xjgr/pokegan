"""Generate Pokémon samples from the trained generator."""

import argparse
import math

MODELO_POR_DEFECTO = "modelo/generador_pokemon_10000_final.h5"
DIM_RUIDO = 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=16,
                        help="how many images to generate (default 16)")
    parser.add_argument("--output", "--salida", dest="salida", metavar="FILE", default="samples.png",
                        help="output PNG file")
    parser.add_argument("--model", "--modelo", dest="modelo", metavar="PATH", default=MODELO_POR_DEFECTO,
                        help="path to the trained generator")
    parser.add_argument("--seed", "--semilla", dest="semilla", metavar="SEED", type=int, default=None,
                        help="seed to reproduce the same result")
    args = parser.parse_args()

    # Se importan aqui dentro para que --help no tenga que cargar TensorFlow,
    # que tarda lo suyo.
    import matplotlib.pyplot as plt
    import tensorflow as tf

    if args.semilla is not None:
        tf.random.set_seed(args.semilla)

    # compile=False: para generar no hace falta reconstruir el optimizador.
    generador = tf.keras.models.load_model(args.modelo, compile=False)

    ruido = tf.random.normal([args.n, DIM_RUIDO])
    imagenes = generador(ruido, training=False)
    # El generador saca tanh, o sea [-1, 1]. Para pintar hace falta [0, 1].
    imagenes = (imagenes + 1) / 2

    lado = math.ceil(math.sqrt(args.n))
    fig = plt.figure(figsize=(lado * 1.5, lado * 1.5))
    for i in range(args.n):
        plt.subplot(lado, lado, i + 1)
        plt.imshow(imagenes[i])
        plt.axis("off")
    plt.tight_layout()
    fig.savefig(args.salida, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"{args.n} images saved to {args.salida}")


if __name__ == "__main__":
    main()
