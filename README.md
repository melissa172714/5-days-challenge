# 5-Days Challenge - Image Restoration with SwinIR

A Streamlit-based web application for image restoration using the SwinIR model.

## How to Add GitHub Copilot Agent in VS Code

Follow these steps to set up and use GitHub Copilot and Copilot Chat (including the Coding Agent) in VS Code:

### Prerequisites

1. **GitHub Account**: You need a GitHub account with access to GitHub Copilot (either through a paid subscription, GitHub Copilot Business, or GitHub Copilot Enterprise)
2. **VS Code**: Make sure you have Visual Studio Code installed (version 1.83 or later recommended)

### Installation Steps

1. **Install GitHub Copilot Extension**:
   - Open VS Code
   - Go to the Extensions view (click the Extensions icon in the Activity Bar or press `Ctrl+Shift+X` / `Cmd+Shift+X`)
   - Search for "GitHub Copilot"
   - Click **Install** on the "GitHub Copilot" extension (published by GitHub)

2. **Install GitHub Copilot Chat Extension**:
   - In the Extensions view, search for "GitHub Copilot Chat"
   - Click **Install** on the "GitHub Copilot Chat" extension (published by GitHub)

3. **Sign In to GitHub**:
   - After installation, you'll see a prompt to sign in to GitHub
   - Click "Sign in to GitHub" and follow the authentication flow in your browser
   - Authorize VS Code to access your GitHub account

4. **Verify Installation**:
   - Once signed in, you should see the Copilot icon in the status bar at the bottom of VS Code
   - Open the Copilot Chat panel by clicking on the chat icon in the Activity Bar or pressing `Ctrl+Shift+I` / `Cmd+Shift+I`

### Using GitHub Copilot Chat (Agent Mode)

1. **Open Copilot Chat**:
   - Click the chat icon in the Activity Bar (left sidebar)
   - Or use the keyboard shortcut `Ctrl+Shift+I` / `Cmd+Shift+I`

2. **Enable Agent Mode**:
   - Open the Copilot Chat panel
   - Look for the **Mode** dropdown at the top of the chat panel
   - Select **"Agent"** from the dropdown menu
   - In Agent mode, Copilot can autonomously make changes to your code, run terminal commands, and iterate on solutions
   
   > **Note**: If you don't see the Agent option, make sure:
   > - Your VS Code and GitHub Copilot extensions are updated to the latest version
   > - You have a GitHub Copilot Pro, Pro+, Business, or Enterprise subscription
   > - You may need to enable it in settings: search for `chat.agent.enabled` and set it to `true`

3. **Using Agent Mode**:
   - Type your request in natural language
   - For example: "Fix the bug in this function" or "Add a new feature to upload images"
   - The agent will analyze your codebase, make edits, and show a step-by-step progress log
   - You can accept, reject, or undo any changes the agent makes

### Keyboard Shortcuts

| Action | Windows/Linux | macOS |
|--------|---------------|-------|
| Open Copilot Chat | `Ctrl+Shift+I` | `Cmd+Shift+I` |
| Inline Chat | `Ctrl+I` | `Cmd+I` |
| Accept suggestion | `Tab` | `Tab` |
| Dismiss suggestion | `Esc` | `Esc` |
| Show next suggestion | `Alt+]` | `Option+]` |
| Show previous suggestion | `Alt+[` | `Option+[` |

### Troubleshooting

- **Not seeing Copilot suggestions?** Make sure you're signed in and your subscription is active
- **Chat not responding?** Try reloading VS Code (`Ctrl+Shift+P` → "Developer: Reload Window")
- **Extension not working?** Check that you have the latest version of VS Code and the Copilot extensions

### More Resources

- [GitHub Copilot Documentation](https://docs.github.com/en/copilot)
- [VS Code Copilot Extension](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot)
- [GitHub Copilot Chat in VS Code](https://docs.github.com/en/copilot/github-copilot-chat/using-github-copilot-chat-in-your-ide)

---

## Application Setup (SwinIR Image Restoration)

### Requirements

```bash
pip install streamlit torch numpy pillow
```

### Running the Application

```bash
streamlit run app.py
```

### Features

- Image super-resolution using SwinIR model
- PSNR and SSIM quality metrics calculation
- Web-based interface for easy image upload and processing
