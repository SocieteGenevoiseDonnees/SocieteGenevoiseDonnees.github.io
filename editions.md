---
title: Editions
layout: support-page
permalink: /editions/
---

{% comment %}
{% include edition-table.html orgs=site.data.editions id="editions" name="Editions" %}
{% endcomment %}

<div id="archives">
{% for category in site.categories %}
  <div class="archive-group">
    {% capture category_name %}{{ category | first }}{% endcapture %}
    <div id="#{{ category_name | slugize }}"></div>
    <p></p>

    <h3 class="category-head">{{ category_name }}</h3>
    <a name="{{ category_name | slugize }}"></a>
    {% assign columns = 4 %}
    {% for post in site.categories[category_name] %}

      <h4><a href="{{ site.baseurl }}{{ post.url }}">{{post.title}}</a></h4> 

    {% endfor %}
  </div>
{% endfor %}
</div>

{% comment %}
<div id="archives">
{% for category in site.categories %}
   {% capture category_name %}{{ category | first }}{% endcapture %}
   {% for post in site.category[category_name] %}
      [{{ post.title }}]({{post.url}})
   {% endfor %}
{% endfor %}
{% endcomment %}

<div id="add-org" class="border-top pt-4 pt-md-6">
  <div class="clearfix gutter-spacious">
    <div class="col-md-6 float-left mb-4">
      <h3 class="alt-h3 mb-2">Create an edition</h3>
      <p class="text-gray">This website is <a href="https://github.com/SocieteGenevoiseDonnees/SocieteGenevoiseDonnees.github.io">open source</a>, therefore anyone in the community can submit edits through pull requests.</p>
      <ol class="text-gray ml-3">
        <li class="mb-2">Click the edit (pencil) icon in the top right corner.</li>
        <li class="mb-2">Add your edition to the list in the appropriate section</li>
        <li class="mb-2">Click "propose file change" at the bottom of the page</li>
        <li class="mb-2">Click "create pull request"</li>
        <li class="mb-2">Provide a brief description of what you're proposing</li>
        <li class="mb-2">Click "Submit pull request"</li>
      </ol>
    </div>

    <div class="col-md-6 float-left">
      <h4 class="mb-2">Guidelines</h4>
      <p class="text-gray">
        While there are many many interesting projects related to data analysis, engineering and machine learning, the above are the only editions we have so far. To add a new edition:
      </p>
      <ul class="mb-4 text-gray ml-3">
        <li>Make sure the topic does not fall under a previous edition</li>
        <li>An introductory blog post should be listed explaining the many facets (or some of them at least) of the topic</li>
        <li>Supply a dataset and/or on notebook/code example that allows others to experience the nature of the topic for themselves</li>
        <li>The topic should be suitable for community discussion. Cutting edge ideas for pushing the boundaries of data science are just as valid as peoples desire to learn about a topic starting from scratch.</li>
      </ul>

  </div>
</div>
