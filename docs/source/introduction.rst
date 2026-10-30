Introduction
============

What is SAETASS?
----------------
SAETASS (Solver for Astroparticle Equation of Transport Analysis in Spherical Symmetry) is a Python-based physics library designed to simulate the transport and acceleration of cosmic rays and other energetic particles in astrophysical environments.

Focusing on one-dimensional spherical symmetry, SAETASS provides robust numerical solvers to evaluate the diffusion-advection transport equation. It is built to seamlessly accommodate complex physical models, including custom spatial diffusion coefficients, advective wind flows, adiabatic energy losses, and localized particle injection sources.

Typical applications include modeling particle transport inside stellar wind bubbles, supernova remnants, and heliospheric-like boundary interactions.

Documentation Overview
----------------------
This documentation is structured to help you get started quickly while providing deep reference material for advanced simulations. You can navigate the documentation using the following sections:

* :doc:`Installation <installation>`: Instructions on how to set up the SAETASS environment and install its dependencies.
* :doc:`API Reference <api/index>`: Detailed technical documentation of the internal modules, classes, and functions.
* :doc:`Tutorials <tutorials/index>`: Easy-to-follow, step-by-step guides covering fundamental concepts. These range from setting up your first spatial grid to running basic simulations.
* :doc:`Examples <examples/index>`: A collection of standalone, reproducible scripts demonstrating specific physical setups and complete analysis.
* :doc:`Terms and conditions <terms>`: License and terms of use for SAETASS.
* :doc:`References <references>`: A bibliography of the academic papers, texts and resources that are referenced in the code and documentation.

Key Features
------------
- **Modular Solvers**: Independent numerical schemes for advection, diffusion, energy losses and source terms, seamlessly integrated via mathematically robust operator splitting parameters and state-of-the-art numerical methods.
- **Flexible Grid Representations**: Dedicated objects for spatial, temporal and momentum matrices to accurately capture complex domain requirements.
- **Physical Accuracy**: Deep integration with `astropy.units` and `astropy.constants` to ensure strict dimensional analysis across all mathematical modules. Every physical input is an Astropy Quantity, validated and converted once to a single set of canonical units at the boundary of the package (see :doc:`the units module <api/units>`).
- **Extensibility**: Designed to be highly modular, allowing researchers to easily plug in custom diffusion coefficients and non-standard advection velocity fields.

Authors and Maintainers
-----------------------
SAETASS is primarily developed and maintained by **José María García Morillo** as a member of the `VHEGA <https://vhega.iaa.es>`_ research group at `Instituto de Astrofísica de Andalucía (IAA-CSIC) <https://www.iaa.csic.es>`_. The author and the group are very much open to feedback, suggestions and contributions; as well as technical or scientific collaborations with the aim of improving the code and its applications.

.. admonition:: José María García Morillo
   :class: note

   .. container:: text-center

      |email| |orcid| |github| |linkedin|

.. |email| image:: https://img.shields.io/badge/Email-jmorillo%40iaa.es-D14836?style=flat-square&logo=minutemailer&logoColor=white
   :target: mailto:jmorillo@iaa.es
   :alt: Email

.. |orcid| image:: https://img.shields.io/badge/ORCID-0009--0008--5232--349X-a6ce39?style=flat-square&logo=orcid&logoColor=white
   :target: https://orcid.org/0009-0008-5232-349X
   :alt: ORCID

.. |github| image:: https://img.shields.io/badge/GitHub-jmgarciamorillo-181717?style=flat-square&logo=github&logoColor=white
   :target: https://github.com/jmgarciamorillo
   :alt: GitHub

.. |linkedin| image:: https://img.shields.io/badge/LinkedIn-José_María_García_Morillo-0A66C2?style=flat-square&logo=linkedin&logoColor=white
   :target: https://www.linkedin.com/in/josem-garcia-morillo/
   :alt: LinkedIn
