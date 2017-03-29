from setuptools import setup

setup(name='pyspeakrec',
    version='0.1',
    description='A Speaker Recognition tool in Python',
    url='https://github.com/AKBoles/pyspeakrec',
    author='Andrew Boles',
    author_email='andrew.boles@my.utsa.edu',
    license='MIT',
    packages=['pyspeakrec'],
    install_requires=['tflearn','librosa','pydub','pyaudio'],
    zip_safe=False)
