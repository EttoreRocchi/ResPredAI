Contributing
============

Thank you for your interest in contributing to ResPredAI! This document provides guidelines for setting up a development environment, running tests, and submitting contributions.


Development Setup
-----------------

.. code-block:: bash

    git clone https://github.com/EttoreRocchi/ResPredAI.git
    cd ResPredAI
    python -m venv .venv
    source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -e .[dev]

Verify installation:

.. code-block:: bash

    respredai --version
    pytest -v -m "not slow"


Running Tests
-------------

**Fast tests (recommended):**

.. code-block:: bash

    pytest -v -m "not slow"

**Full suite:**

.. code-block:: bash

    pytest -v


Making Changes
--------------

1. Create a branch
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git checkout -b feature/name
    # or
    git checkout -b fix/issue

2. Implement changes
~~~~~~~~~~~~~~~~~~~~

- Add/update tests if needed
- Update docs if needed

3. Validate
~~~~~~~~~~~

.. code-block:: bash

    pytest -v -m "not slow"
    respredai validate-config example/config_example.ini --check-data

4. Commit
~~~~~~~~~

.. code-block:: bash

    git commit -m "[type] Description"

Types: ``feat``, ``fix``, ``enh``, ``docs``, ``test``, ``refactor``, ``chore``


Submitting a Pull Request
-------------------------

.. code-block:: bash

    git push origin feature/name

Then open a PR on GitHub:

- Describe your changes
- Reference issues
- Ensure CI passes


Need Help?
----------

- Open an issue on GitHub
- Check existing docs & issues


License
-------

By contributing to ResPredAI, you agree that your contributions will be licensed under the MIT License.
