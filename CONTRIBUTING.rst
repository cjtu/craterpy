How to contribute to craterpy
=============================
Thank you for your interest in contributing to craterpy!

Who can contribute?
-------------------
*You!* Bug reports, user feedback, and feature requests are very real contributions and begin over on the `issues`_ board.

To contribute to the code directly, even if it's your first time contributing to open source, see below.

Where to start?
---------------
A good place to start is by scanning the open `issues`_ and seeing if there is something that you are interested in taking on. Drop a comment on an issue to let us know that you want to work on it, or leave a question if something needs to be clarified.

.. _`issues`: https://github.com/cjtu/craterpy/issues

How to get started?
-------------------
You will need to download the repo and install the development environment locally to start contributing code.

1. Fork the `main repository`_ by clicking the link and clicking the Fork button in the upper right corner.::

2. Clone the forked repository to your machine. Check out `GitHub Desktop`_ if it's your first time using Git. ::

3. Install uv from https://docs.astral.sh/uv/#installation then download dependencies using uv::

    uv sync

4. Make sure the package installed correctly::

    uv run pytest craterpy

5. Make a new git branch::

    git checkout -b feature-xyz

6. Hack away!

Preparing a commit
------------------

When you are ready to contribute your changes, make sure to:

1. Add unittests to the tests folder.::

2. Test your changes::

    uv run pytest craterpy

3. Add and commit your changes::
    
    git add . 
    git commit -m "Descriptive commit message here" 

4. The `ruff` pre-commit hook may reformat some of your files to ensure a standard python style. If changes are made, repeat step 3 to add and commit your changes.::

5. Push your branch and open a Pull Request on GitHub!

.. _`main repository`: https://github.com/cjtu/craterpy
.. _`Github Desktop`: https://github.com/apps/desktop
.. _`GitHub Workflow`: https://docs.github.com/en/get-started/using-github/github-flow


Thanks fro contributing!
------------------------
Sit tight while a collaborator reviews your contribution. We might ask you to clarify some code or add tests and will work with you to get your contribution merged!

Stuck? Having doubts about getting started?
-------------------------------------------
Get in touch on the Issue board or by email. We'd be happy to help you get started!

Code of Conduct
---------------
This community is governed by a `code of conduct`_. This is an inclusive community and attitudes or behaviours that make other members feel unsafe or uncomfortable will not be tolerated.

.. _`code of conduct`: https://github.com/cjtu/craterpy/blob/master/CODE_OF_CONDUCT.md

Happy contributing and have a great day!
========================================
