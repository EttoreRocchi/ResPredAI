Installation
============

Install from PyPI:

.. code-block:: bash

    pip install respredai

Or install from source:

.. code-block:: bash

    git clone https://github.com/EttoreRocchi/ResPredAI.git
    cd ResPredAI
    # For development (includes pytest)
    pip install -e ".[dev]"

Optional: TabPFN support
------------------------

TabPFN is shipped as an optional extra to keep the base install lean. Install
it only if you plan to use the ``TabPFN`` model:

.. code-block:: bash

    pip install "respredai[tabpfn]"

TabPFN v3 also requires a `PriorLabs API token <https://priorlabs.ai/docs>`_.
Export it before running any pipeline that includes ``TabPFN`` in
``[Pipeline] models``:

.. code-block:: bash

    export TABPFN_TOKEN="your-token-here"

If `TabPFN` is requested without the extra installed or without the token set,
ResPredAI fails at the start of the run.

Testing the Installation
------------------------

Verify the installation:

.. code-block:: bash

    respredai --version
