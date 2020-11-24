sid
===

sid is a simulator for infectious diseases. It combines features of a prototypical
Susceptible-Exposed-Infected-Recovered (SEIR) model and an agent based simulation model.

sid's focus is on predicting the effects of policies on the spread of an infectious
disease. To accomplish this task it is built to capture important aspects of human
contacts. In particular, sid allows:

- Recurrent contacts (e.g. school mates meeting Monday through Friday).
- Different types of contacts (e.g. work and leisure contacts).
- Varying number of contacts, infection probabilities, matching patterns and disease
  progressions according to individual characteristics such as age and gender.

The documentation is structured into four parts.

.. raw:: html

    <div class="row">
        <div class="column">
            <a href="tutorials/index.html" id="index-link">
                <div class="card">
                    <img src="_static/images/light-bulb.svg"
                         alt="tutorials-icon" height="52"
                    >
                    <h5>Tutorials</h5>
                    <p>
                        Tutorials help you to get started with sid. Here you learn
                        everything about the interface and the capabilities.
                    </p>
                </div>
            </a>
        </div>
        <div class="column">
            <a href="how_to_guides/index.html" id="index-link">
                <div class="card">
                    <img src="_static/images/book.svg"
                         alt="how-to guides icon" height="52"
                    >
                    <h5>How-to Guides</h5>
                    <p>
                        How-to guides are designed to provide detailed instructions for
                        very specific and advanced tasks.
                    </p>
                </div>
            </a>
        </div>
        <div class="column">
            <a href="explanations/index.html" id="index-link">
                <div class="card">
                    <img src="_static/images/books.svg"
                         alt="explanations icon" height="52"
                    >
                    <h5>Explanations</h5>
                    <p>
                        Explanations give detailed information to key topics and
                        concepts which underlie the package.
                    </p>
                </div>
            </a>
        </div>
        <div class="column">
            <a href="reference_guides/index.html" id="index-link">
                <div class="card">
                    <img src="_static/images/coding.svg"
                         alt="reference guides icon" height="52"
                    >
                    <h5>Reference Guides</h5>
                    <p>
                        Reference Guides can be read alongside the source code for a
                        better understanding.
                    </p>
                </div>
             </a>
        </div>
    </div>


.. toctree::
    :hidden:

    tutorials/index
    how_to_guides/index
    explanations/index
    reference_guides/index
