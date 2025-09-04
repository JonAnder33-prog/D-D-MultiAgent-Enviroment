# Simulación y optimización de Agentes Inteligentes que juegan a Dragones y Mazmorras

Dungeons & Dragons es el juego de rol de mesa más popular del mundo, con millones de jugadores en todo el planeta. Este juego, que combina narrativa, interpretación y estrategia, requiere la presencia de un grupo de jugadores y de un jugador que será la figura central conocida como Dungeon Master (DM), responsable de diseñar y dirigir la partida.

Uno de los principales retos a los que se enfrenta el DM consiste en crear combates que sean lo suficientemente desafiantes como para resultar interesantes, pero sin llegar a ser muy difíciles para los jugadores. La falta de métodos y herramientas para estimar la dificultad de un enfrentamiento de antemano de forma precisa y eficaz convierte esta tarea en un proceso complejo y, en gran medida, dependiente de la intuición y la experiencia del DM.

Debido a ello, en este TFM se ha desarrollado un sistema basado en el aprendizaje por refuerzo. Dicho sistema emplea agentes inteligentes capaces de controlar a cada uno de los personajes del combate. Una vez entrenados dichos agentes y mediante la ejecución de varios episodios se puede estimar la dificultad del combate de manera más precisa, rápida y eficaz mediante el cálculo de las probabilidades de supervivencia de cada personaje en una serie de simulaciones.

Para realizar pruebas con el código es necesario tener instalado en el ordenador la libreria Rllib con la versión 2.44.1, debido a que es la versión que menos problemas de compatibilidad tienen con los sistemas operativos. 

Una vez instalada la librería, hay que descargar el archivo Entorno_DND.py que se compone del entorno que simula los combates de D&D junto con las funciones que crean y entrenan a los agentes inteligentes para que aprendan a controlar a sus respectivos personajes. Ejecutando dicho el codigo se entrenarán a los agentes para que controlen a los personajes establecidos en el codigo.
