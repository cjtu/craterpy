How to contribute to craterpy
=============================
Thank you for your interest in contributing to craterpy!

Who can contribute?
-------------------
*Anyone!* This project is as much a tool to learn about open source science as it is a tool to improve planetary data science workflows. If you've never contributed to open source before, great! Drop us a line! If you're an expert hacker and think we're doing everything wrong, awesome! We want to hear from you too!

Why contribute?
---------------
If you're interested in any of the following:

- planets
- science
- open source
- contributing to active research
- learning in a supportive environment

Or you already use craterpy and want to make it better for everyone!

Where to start?
---------------
A good place to start is by scanning the `issue tracker`_ and seeing if there is something that you are interested in taking on. Drop a comment on an issue to let us know that you want to work on it, or leave a question if something needs to be clarified.

.. _`issue tracker`: https://github.com/cjtu/craterpy/issues

How to get started?
-------------------
If you made it this far and you're still interested, that's great! Here are four simple steps to get started contributing to craterpy:

1. Fork the `main repository`_ by clicking the link and clicking the Fork button in the upper right corner (read about `forking`_).

2. Clone the forked repository to your machine (Git must be `installed`_). ::

    git clone https://github.com/your_username/craterpy.git
    cd craterpy

3. Make the conda-forge channels available ::

   conda config --add channels conda-forge

4. Set up a conda virtual environment from the ``.environment.yml`` (requires `Anaconda`_). ::

    conda env create -f .environment.yml

   on Windows::

    activate craterpy-dev
    python setup.py -q install

   on UNIX/Mac users::

    source activate craterpy-dev
    python setup.py -q install

5. Make a feature branch to be your working branch. Choose any name you like. ::

    git checkout -b feature-xyz

Now that your dev environment is set up, you can start making contributions! If you haven't already, check out the `issue tracker`_ for suggestions on where to get started.

.. _`main repository`: https://github.com/cjtu/craterpy
.. _`forking`: https://guides.github.com/activities/forking/
.. _`installed`: https://git-scm.com/downloads
.. _`Anaconda`: https://www.anaconda.com/download/

When and how to submit contributions?
-------------------------------------
Contributions are merged into the master branch via the dreaded **pull request**. These aren't actually that scary, and will go smoothest if you can answer yes to all of the following questions:

- Is my contribution small and self-contained?

- Does my contribution pass all unittests when I run ``py.test`` from the root directory?

- Does my contribution pass all style tests when I run ``flake8``?

- If my contribution adds new class(es), function(s) and/or method(s), do they all have descriptive docstrings?

- If my contribution adds new functionality, did I add good test cases to the ``tests/`` folder? Do these pass ``py.test --cov`` with close to 100% coverage?

If you answered yes to all of the above, you're probably good to go! Read the next section and open a Pull Request!

Submitting a Pull Request
-------------------------
Three simple steps:

1. When you are happy with your contributions, commit them to your feature branch and then push them to your forked repository (if you run into trouble, check out this great guide to `git commands`_).

2. Go to `pulls <https://github.com/cjtu/craterpy/pulls>`_ and click the "New pull request" button.

3. Click "compare across forks", then use the drop-downs to choose:

   - `base fork : cjtu.craterpy`
   - `base : master`
   - `head fork : your-user/craterpy`
   - `compare : your-feature-branch`.

4. Click "Create pull request" and describe your contribution in the box provided. If you are addressing an issue on the `issue tracker`_, reference it by number in the pull request title to auto reference it (e.g. "Solves #12" would be a great title if you solved a bug detailed in issue 12).

.. _`git commands`: http://git.huit.harvard.edu/guide/

You did it!
-----------
At this point, you've sucessfully submitted a pull request to craterpy! Sit tight while a collaborator reviews your contribution. We might ask you to clarify or update something, but if you followed this guide then we should be able to merge your contribution in no time!

What if I'm stuck on something or having doubts about getting started?
----------------------------------------------------------------------
Email Christian at cj.taiudovicic@gmail.com. Seriously. If you made it this far, we want to hear from you!

Code of Conduct
---------------
This community is governed by a `code of conduct`_. This is an inclusive community and attitudes or behaviours that make other members feel unsafe or uncomfortable will not be tolerated.

.. _`code of conduct`: https://github.com/cjtu/craterpy/blob/master/CODE_OF_CONDUCT.rst

Happy contributing and have a great day!
========================================
