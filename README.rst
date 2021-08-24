=============
Data-Xray
=============

Data-Xray provides a Pythonic interface to organization and analysis of
data, acquired primarily with Nanonis controllers, manufactured by Specs GmbH.

The primary emphasis at the moment is on scanning probe microscopy measurements,
but most of the analysis applies to other domains of hyperspectral data.

Data organization is achieved through the use of xarray structure - which is 
an ingenious extension of Pandas phenomenology onto arbitrary dimensions.

Data-Xray draws upon many truly excellent python libraries, for which we as developers
are phenomenally greatful. 

Installation / Setup
********************

Use pip to install the library:

    pip install data-xray
    

**Testing**

There are a few ways to run the unit tests.

One option is to use the shell script in the root of the repository
called *test.example.sh*. Copy it using ``cp test.example.sh test.sh``.
Edit *test.sh* to include your Cicero API username and password. Then, run
the tests using ``./test.sh``.

Another option is to edit the ``test/tests.py`` file directly, adding your
Cicero API credentials where indicated. Doing so will allow you to execute
tests using ``nosetests`` (if you have the nose package installed), or
using ``python setup.py test``, or invoking the ``tests.py`` file itself.

Documentation
*************



Help!
*****

License
*******

data-xray is licensed under the Apache 2.0 license. See ``LICENSE.txt`` for
more details.

Contribute
**********

See a bug? Want to improve the docs or provide more examples? Thank you!
Please open a pull-request with your improvements and we'll work to respond
to it in a timely manner.
