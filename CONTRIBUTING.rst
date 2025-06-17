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

3. Install the dependencies using poetry::

    poetry install

4. Make sure the package installed correctly::

    poetry run pytest craterpy

5. Make a new git branch::

    git checkout -b feature-xyz

6. Hack away!

Preparing a commit
------------------

When you are ready to contribute your changes, make sure to:

1. Add unittests to the tests folder.

2. Test your changes::

    poetry run pytest craterpy

3. Apply automatic code formatting with black::

    poetry run black craterpy

4. Run pylint to check your code style and fix any errors::

    poetry run pylint craterpy

8. If linting and tests look good locally, add and commit your changes::
    
    git add . 
    git commit -m "Descriptive commit message here" 

Now you're ready to push your changes and open a Pull Request on GitHub!

.. _`main repository`: https://github.com/cjtu/craterpy
.. _`forking`: https://guides.github.com/activities/forking/
.. _`installed`: https://git-scm.com/downloads
.. _`Anaconda`: https://www.anaconda.com/download/


Submitting a Pull Request
-------------------------
Three simple steps:

1. When you are happy with your local commit, push them to your forked repository (if you run into trouble, check out this great guide to `git commands`_)::

    git push
    # Or if this is your first time pushing a branch:
    git push --set-upstream origin <branch-name>

2. Go to the `Pull requests <https://github.com/cjtu/craterpy/pulls>`_ tab on GitHub and click the "New pull request" button.

3. Click "compare across forks", then use the drop-downs to choose:

   - `base fork : cjtu.craterpy`
   - `base : master`
   - `head fork : your-user/craterpy`
   - `compare : your-feature-branch`.

4. Click "Create pull request" and describe your contribution in the box provided. If you are addressing an issue on the `issue tracker`_, reference it by number in the pull request title to auto reference it (e.g. "Solves #12" would be a great title if you solved a bug detailed in issue 12).

.. _`git commands`: http://git.huit.harvard.edu/guide/

You did it!
-----------
At this point, you've successfully submitted a pull request to craterpy! Sit tight while a collaborator reviews your contribution. We might ask you to clarify or update something, but if you followed this guide then we should be able to merge your contribution in no time!

What if I'm stuck on something or having doubts about getting started?
----------------------------------------------------------------------
Get in touch on the Issue board or by emailing Christian at cj.taiudovicic@gmail.com. We'd be happy to help you get started!

Code of Conduct
---------------
This community is governed by a `code of conduct`_. This is an inclusive community and attitudes or behaviours that make other members feel unsafe or uncomfortable will not be tolerated.

.. _`code of conduct`: https://github.com/cjtu/craterpy/blob/master/CODE_OF_CONDUCT.md

Happy contributing and have a great day!
========================================
