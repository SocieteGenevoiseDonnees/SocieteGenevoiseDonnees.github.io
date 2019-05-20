# SGD [![Build Status](https://travis-ci.org/SocieteGenevoiseDonnees/SocieteGenevoiseDonnees.github.io.svg?branch=gh-pages)](https://travis-ci.org/SocieteGenevoiseDonnees/SocieteGenevoiseDonnees.github.io)

The site is open source (here's all the code!) and is a tool _for_ and _by_ the community.

### Under the Hood

This site is made with [Jekyll](http://jekyllrb.com), an open source static site generator. This means the Jekyll program takes the content we want to be on the site and turns them into HTML files ready to be hosted somewhere. Awesomely, GitHub provides free web hosting for repositories, called [GitHub Pages](http://pages.github.com/), and that's how this site is hosted.

## Contributing

#### Fix/Edit Content

If you see an error or a place where content should be updated or improved, just fork this repository to your account, make the change you'd like and then submit a pull request. If you're not able to make the change, file an [issue](https://github.com/SocieteGenevoiseDonnees/SocieteGenevoiseDonnees.github.io/issues/new).

---

## To Set up Locally

You can take all the files of this site and run them just on your computer as if it were live online, only it's just on your machine.

#### Requirements

* [Jekyll](http://jekyllrb.com/)
* [Ruby](https://www.ruby-lang.org/en/)
* [Git](http://git-scm.com/)

_If you have installed [GitHub Desktop](https://desktop.github.com), Git was also installed automatically._

To copy the repository's files from here onto your computer and to view and serve those files locally, at your computer's command line type:

```bash
git clone https://github.com/SocieteGenevoiseDonnees/SocieteGenevoiseDonnees.github.io.git -b master
cd SocieteGenevoiseDonnees.github.io.git
script/bootstrap
script/server
```
Open `http://localhost:4000` in your browser

## Contributing

images should be placed in `/assets/`

### editions
new editions are placed in _posts the core *blog* post introducing the topic needs the tag category in the front matter, sub pages cannot have this attribute or else they will show up on the main page. To link between files merely add `{% page_url the-name-of-the-markdown-file %}.

#### Jupyter Notebooks

Download jupyter notebooks as markdown, add the front matter to the top (no category). Images can be placed in the `/assets/` directory with a folder relating to the edition e.g. `/assets/StepFive-Plotting/` then this prefix is manually added to all image links in markdown. 

** to do **
- html tables do not render properly leading to some very ugly pandas
- no support for mathjax yet
