===
sid
===

sid is a **s**\ imulator for **i**\ nfectious **d**\ iseases. It combines features of a
prototypical Susceptible-Exposed-Infected-Recovered (SEIR) model and an agent based
simulation model.

sid's focus is on predicting the effects of policies on the spread of an infectious
disease. To do this it is built to capture important aspects of human contacts.
In particular, sid allows:

- recurrent contacts, such as school mates meeting Monday through Friday.
- different types of contacts, such as work and leisure contacts.
- varying number of contacts, infection probabilities, matching patterns and disease
    progression according to individual characteristics such as age and gender.

The documentation is structured into four parts.

.. raw:: html

    <div class="container" id="index-container">
        <div class="row">
            <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex">
                <a href="tutorials/index.html" id="index-link">
                    <div class="card text-center intro-card shadow">
                        <img src="_static/images/light-bulb.svg" class="card-img-top"
                             alt="tutorials-icon" height="52"
                        >
                        <div class="card-body flex-fill">
                            <h5 class="card-title">Tutorials</h5>
                            <p class="card-text">
                                Tutorials help you to get started with sid. Here you
                                learn everything about the interface and how to
                                accomplish basic tasks.
                            </p>
                        </div>
                    </div>
                </a>
            </div>
            <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex">
                <a href="explanations/index.html" id="index-link">
                    <div class="card text-center intro-card shadow">
                        <img src="_static/images/books.svg" class="card-img-top"
                             alt="explanations icon" height="52"
                        >
                        <div class="card-body flex-fill">
                            <h5 class="card-title">Explanations</h5>
                            <p class="card-text">
                                Explanations give detailed information to key topics and
                                concepts which underlie the package.
                            </p>
                        </div>
                    </div>
                </a>
            </div>
            <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex">
                <a href="reference_guides/index.html" id="index-link">
                    <div class="card text-center intro-card shadow">
                        <img src="_static/images/coding.svg" class="card-img-top"
                             alt="reference guides icon" height="52"
                        >
                        <div class="card-body flex-fill">
                            <h5 class="card-title">Reference Guides</h5>
                            <p class="card-text">
                                Reference Guides explain the implementation. If you are
                                interested in the inner workings, you will find this
                                section helpful.
                            </p>
                        </div>
                    </div>
                 </a>
            </div>
        </div>
    </div>

.. toctree::
    :hidden:

    tutorials/index
    explanations/index
    reference_guides/index


.. toctree::
   :maxdepth: 1
   :caption: Other Topics

   api
