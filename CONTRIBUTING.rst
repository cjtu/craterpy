Contibuting to craterpy
=======================

Questions about contributing can be sent to Christian at cj.taiudovicic@gmail.com. This file is largely modeled after scikit-learn's `CONTRIBUTING <https://github.com/scikit-learn/scikit-learn/blob/master/CONTRIBUTING.md>`_.

How to Contribute
-----------------
The preferred workflow for contributing is to fork the `main repository <https://github.com/cjtu/craterpy>`_ on GitHub, clone, and develop on a branch. When you are ready, submit a pull request to the main repository. Steps:

1. Fork the `project repository <https://github.com/cjtu/craterpy>`_ by clicking on the 'Fork' button near the top right of the page. This creates a copy of the code under your GitHub user account. For more details on how to fork a repository see `this guide <https://help.github.com/articles/fork-a-repo/>`_.

2. Clone your fork of the craterpy repo from your GitHub account to your local disk::

   $ git clone git@github.com:YourLogin/craterpy.git
   $ cd craterpy

3. Create a ``feature`` branch to hold your development changes::

   $ git checkout -b my-feature

   Always use a ``feature`` branch. It's good practice to never work on the ``master`` branch!

4. Develop the feature on your feature branch. Add changed files using ``git add`` and then ``git commit`` files::

   $ git add modified_files
   $ git commit

   to record your changes in Git, then push the changes to your GitHub account with::

   $ git push -u origin my-feature

5. Follow `these instructions <https://help.github.com/articles/creating-a-pull-request-from-a-fork>`_ to create a pull request from your fork. This will send an email to the committers.

(If any of the above is unfamiliar to you, please look up the
`Git documentation <https://git-scm.com/documentation>`_, or ask a friend or another contributor for help.)

Pull Request Checklist
----------------------
Please check that your contribution complies with the following rules before submitting a pull request:

-  If your pull request addresses an issue, mention the issue number in the pull request description. This will make sure a link back to the original issue is created.

-  All methods should have informative docstrings with sample usage presented as doctests when appropriate.

- New functionality should add (and pass) high coverage test cases to the ``tests/`` folder. See included tests for reference.

-  When adding additional functionality, provide an example script in the ``examples/`` folder. Have a look at other examples for reference.

Filing bugs
-----------
We use GitHub issues to track all bugs and feature requests; feel free to open an issue if you have found a bug or wish to see a feature implemented.

Please check that your issue complies with the following rules before submitting:

-  Verify that your issue is not being currently addressed by other
   `issues <https://github.com/cjtu/craterpy/issues?q=>`_ or `pull requests <https://github.com/cjtu/craterpy/pulls?q=>`_.

-  Please ensure all code snippets and error messages are formatted in
   appropriate code blocks. See `Creating and highlighting code blocks <https://help.github.com/articles/creating-and-highlighting-code-blocks>`_.

New contributor tips
--------------------

A great way to start contributing to craterpy is to pick an item from the list of `Easy issues <https://github.com/cjtu/craterpy/labels/easy>`_ in the issue tracker. Resolving these issues allow you to start contributing to the project without much prior knowledge.
