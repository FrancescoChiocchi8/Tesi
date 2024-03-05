from setuptools import setup

setup(
    name='Tesi',
    version='',
    packages=['Tesi', 'Tesi.scripts', 'Tesi.scripts.cgr', 'Tesi.scripts.cnn', 'Tesi.scripts.file_handling',
              'Tesi.scripts.hyperparameter_optimization'],
    url='https://github.com/FrancescoChiocchi8/Tesi',
    license='MIT License',
    author='Francesco Chiocchi',
    author_email='francesco.chiocchi@studenti.unicam.it',
    description='Reti convoluzionali e chaos game representation per la classificazione di sequenze di RNA'
)
