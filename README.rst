pyspeakrec
==========

Python app that lets you create, use and edit a functioning deep
learning speaker recognition system.

Getting Started
---------------

These instructions will get you a copy of the project up and running on
your local machine for development and testing purposes. See deployment
for notes on how to deploy the project on a live system.

Installation
~~~~~~~~~~~~

For use in Python programs in Linux, install using ``pip``.

Linux
^^^^^

::

    $ pip install pyspeakrec

Running the tests
-----------------

Explain how to run the automated tests for this system

Break down into end to end tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Explain what these tests test and why

::

    To be done.

Use as End-to-End
-----------------

The following describes installation instructions for both Mac and
Windows systems for use as a complete package.

Mac (UNIX)
^^^^^^^^^^

::

    $ git clone https://github.com/AKBoles/pyspeakrec.git
    $ cd pyspeakrec/
    $ pip install -r setup.txt --user

Windows 10
^^^^^^^^^^

First, check installation of python on Windows 10 by opening up a
command prompt and typing ``python`` and seeing which version is
installed. If it is ``Python3.5.2`` then you can continue the
installation. If it is not, follow instructions on how to upgrade python
on a Windows system, found at
`Python3.5-Windows <https://github.com/AKBoles/Installation-Documentation/blob/master/Python3-Windows10.md>`__.

Next, install Git for Windows by following the instructions on
`git-scm <https://git-scm.com/>`__.

Following this installation launch ``Git bash`` and clone this
repository by typing the following command:

::

    $ git clone https://github.com/AKBoles/pyspeakrec.git
    $ cd pyspeakrec/
    $ pip install -r setup.txt --user

Troubleshooting
~~~~~~~~~~~~~~~

Mac (UNIX)
^^^^^^^^^^

If problems occur with the Mac installation, run the following commands
to try and fix it.

::

    $ xcode-select --install
    $ pip install --global-option='build_ext' --global-option='-I/usr/local/include' --global-option='-L/usr/local/lib' pyaudio --user
    $ pip install librosa --user
    $ brew install portaudio

If the installation of ``portaudio`` gives an error, install from source
by doing:

::

    $ git clone https://github.com/jleb/pyaudio.git
    $ cd pyaudio/
    $ python setup.py install

Additionally, check for updates (and update if there are) in ``pip`` by
using:

::

    $ python -m pip install --upgrade pip

Windows 10
^^^^^^^^^^

Check for updates (and update if there are) in ``pip`` by using:

::

    $ python -m pip install --upgrade pip

Version
-------

Current version: ``0.1``

Authors
-------

-  Andrew K. Boles

License
-------

This project is licensed under the MIT License - see the
`LICENSE.txt <LICENSE.txt>`__ file for details

Acknowledgments
---------------

-  Inspiration to pursue this type of work came from
   `Pannous <https://github.com/pannous>`__
