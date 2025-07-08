# Guide: Setting up Slack integration and credentials

1. **Create a Slack App and Bot Token:**

   - Go to https://api.slack.com/apps and create a new app.
   - Add the "chat:write" permission under OAuth & Permissions.
   - Install the app to your workspace and copy the **Bot User OAuth Token** (starts with `xoxb-`).

2. **Find your Slack Channel ID:**

   - Open Slack, go to the desired channel.
   - Click the channel name, then "Copy Channel ID" (or find it in the URL as the last part after `/messages/`).

3. **Create/Edit `credentials.yaml`:**

   - Copy `credentials.yaml` from the sample provided.
   - Fill in your HuggingFace token, Slack Bot Token, and Slack Channel ID.
   - Example:
     ```yaml
     HF_TOKEN: "your_huggingface_token_here"
     OUTPUT_YAML: "output.yaml"
     SLACK_BOT_TOKEN: "xoxb-..."
     SLACK_CHANNEL: "C1234567890"
     ```

4. **Troubleshooting:**

   - If Slack notifications fail, check that your bot is invited to the channel and has the correct permissions.
   - You can invite the bot by typing `/invite @your-bot-name` in the channel.

---

For further help, see the [Slack API documentation](https://api.slack.com/).
