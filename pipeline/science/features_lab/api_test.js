// Load environment variables
import dotenv from "dotenv";
dotenv.config();

// Import OpenAI package
import OpenAI from "openai";

// Main function to test Azure OpenAI
const main = async () => {
  try {
    console.log("Configuring OpenAI with Azure...");
    
    // Print environment variables for debugging (sanitized)
    console.log(`API Key exists: ${!!process.env.AZURE_OPENAI_API_KEY_BACKUP}`);
    console.log(`API Endpoint exists: ${!!process.env.AZURE_OPENAI_ENDPOINT_BACKUP}`);
    
    // Create an instance of the OpenAI client configured for Azure
    const openai = new OpenAI({
      apiKey: process.env.AZURE_OPENAI_API_KEY_BACKUP,
      baseURL: `${process.env.AZURE_OPENAI_ENDPOINT_BACKUP}/openai/deployments/gpt-4o`,
      defaultQuery: { "api-version": "2024-06-01" },
      defaultHeaders: { "api-key": process.env.AZURE_OPENAI_API_KEY_BACKUP }
    });

    console.log("OpenAI client with Azure configuration initialized successfully");

    // Define the user prompt
    const prompt = "what is 2 + 2 = ? explain the answer";
    console.log("Executing chat completion with prompt:", prompt);
    
    // Call the chat completions API
    const completion = await openai.chat.completions.create({
      messages: [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: prompt }
      ],
      model: "gpt-4o", // This is ignored for Azure OpenAI as it's specified in the URL
      temperature: 0,
      max_tokens: 500
    });
    
    // Print the response
    console.log("Response:");
    console.log(completion.choices[0]?.message?.content || "No response content received.");
  } catch (error) {
    console.error("Error:", error.message);
    console.error("Details:", error);
  }
};

main();
