# Hoodie

## 🚀 About Hoodie

Hoodie is a gatsby theme created for writing tech blogs. Markdown, Code Highlighting in various programming languages, and Katex syntax are supported. Also, you can easily categorize articles into tags and series.

Start your blog with a neatly designed Hoodie that supports dark mode.

## Features

- Markdown
- Code Highlighting
- Katex Syntax
- Dark Mode (Responsive to the settings of the OS)
- Tag Categoriazation
- Series Categorization
- Responsive Web
- SEO
- Utterance (Comment widget)

Getting started your blog with Hoodie by following steps below. It's very easy 😉.

## 1. Create a Gatsby site

> Make sure you have **node.js** installed on your computer.

```
$ npx gatsby new my-hoodie-blog https://github.com/devHudi/Hoodie
```

## 2. Start dev server

```
$ cd my-hoodie-blog
$ npm run start
```

Now you can access to your blog at localhost:8000.

## 3. Create your own Github repository

Utterance comment widget is based on **Github issue system**. So you need your own GitHub repository. Also, if you want to publish your blog through Github Pages or Netlify, the Github Repository is a necessary.

If you don't know how to create a GitHub repository, follow the [official GitHub documentation](https://docs.github.com/en/github/getting-started-with-github/create-a-repo).

### Add remote repository

```
git remote add origin https://github.com/{YOUR_GITHUB_NAME}/{YOUR_REPOSITORY_NAME}
```

## 4. Write blog-config.js

```javascript
module.exports = {
  title: "Hoodie.gatsby",
  description: "Hello :) I'm Hudi who developed Hoodie theme.",
  author: "Hudi",
  siteUrl: "https://hudi.blog",
  links: {
    github: "https://github.com/devHudi",
    facebook: "https://www.facebook.com/profile.php?id=100057724153835",
    instagram: "https://www.instagram.com/dawn_fromeast/",
    etc: "https://www.notion.so/Hudi-s-R-sum-0c1d1a1b35284d1eaf05c5bfac4a3cad",
  },
  utterances: {
    repo: "devHudi/Hoodie",
    type: "pathname",
  },
}
```

Hoodie provides a configuration file called `blog-config.js`. In this file, you can configure blog, biography (profile), and utterance. The website settings you are currently viewing are as above.

Configure `blog-config.js` to suit your blog. However, it is recommended not to modify `utterances.type`.

## 5. Add your content

Markdown content is in `contents/posts`. You can write and add your articles like the sample posts here. [Click here](/writing-guide) to see the detail writing guide.

## 6. Deploy your blog

### 6-1 via Netlify

Follow the Connecting to Netlify steps in [A Step-by-Step Guide: Gatsby on Netlify](https://www.netlify.com/blog/2016/02/24/a-step-by-step-guide-gatsby-on-netlify/). It's not difficult.

If you connect the github repository using Netlify, it is automatically distributed whenever you push it, so it is convenient.

### 6-2. via Github Pages

#### Case 1

If the repository name is the same as your GitHub name (if your GitHub page URL is `https://{YOUR_GITHUB_NAME}.github.io`) run the following command to deploy.

```
$ npm run deploy-gh
```

#### Case 2

If the repository name is different from your GitHub name (if your GitHub page URL is `https://{YOUR_GITHUB_NAME}.github.io/{YOUR_REPOSITORY_NAME}`) run the following command to deploy.

```
$ npm run deploy-gh-prefix-paths
```

In the above case, you need to change `pathPrefix` in `gatsby-config.js` to your repository name.

### 6-3. other platforms

```
$ npm run build
```

You can build the gatsby website with the command above. The build output is created in the `/public` directory. Deploy the `/public` directory using the command for the platform you want to deploy.

## 7. Cutomize

### Project Structure

You can customize your own Hoodie by referring to the following file structure 🙊.

```
├── node_modules
├── contents
│   └── posts // your articles are here
├── public // build outputs are here
└── src
    ├── assets
    │   └── theme // theme config is here
    ├── components
    │   └── Article
    │       └── Body
    │           └── StyledMarkdown
    │               └── index.jsx // markdown styles are here
    │   ...
    ├── fonts // webfonts are here
    ├── hooks
    ├── images
    ├── pages // page components are here
    ├── reducers
    ├── templates // post components are here
    └── utils
```
