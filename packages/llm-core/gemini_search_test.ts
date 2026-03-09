import { b } from 'llm-core';
import { Collector } from '@boundaryml/baml';

console.log("Testing Gemini with Google Search grounding...\n");

const collector = new Collector();

console.log("With search:");
const resp = await b.GeminiSearchTest({ collector });

console.log("\nGenerated Response:");
console.log(resp);
console.log("\n" + "=".repeat(80));

// Access the raw HTTP response to get grounding metadata
const lastLog = collector.last;
if (lastLog) {
    console.log("\nFunction Log:");
    console.log("- Function:", lastLog.functionName);
    console.log("- Timing:", lastLog.timing.toString());
    console.log("- Usage:", lastLog.usage);

    const calls = lastLog.calls;
    if (calls && calls.length > 0) {
        console.log("\n" + "=".repeat(80));
        console.log("LLM Call Details:");
        const call = calls[0];
        console.log("- Provider:", call.provider);
        console.log("- Client:", call.clientName);

        const httpResponse = call.httpResponse;
        if (httpResponse) {
            console.log("\nHTTP Response:");
            console.log("- Status:", httpResponse.status);

            try {
                const body = httpResponse.body.json();
                console.log("\nFull Response Body:");
                console.log(JSON.stringify(body, null, 2));

                // Extract grounding metadata if present
                if (body.candidates && body.candidates[0]?.groundingMetadata) {
                    const groundingMetadata = body.candidates[0].groundingMetadata;
                    console.log("\n" + "=".repeat(80));
                    console.log("GROUNDING METADATA:");
                    console.log(JSON.stringify(groundingMetadata, null, 2));

                    if (groundingMetadata.webSearchQueries) {
                        console.log("\n" + "=".repeat(80));
                        console.log("Search Queries Used:");
                        groundingMetadata.webSearchQueries.forEach((q: string, i: number) => {
                            console.log(`  ${i + 1}. ${q}`);
                        });
                    }

                    if (groundingMetadata.groundingChunks) {
                        console.log("\n" + "=".repeat(80));
                        console.log("Sources:");
                        groundingMetadata.groundingChunks.forEach((chunk: any, i: number) => {
                            console.log(`  [${i + 1}] ${chunk.web?.title || 'Unknown'}`);
                            console.log(`      ${chunk.web?.uri || 'No URL'}`);
                        });
                    }
                }
            } catch (e) {
                console.error("Failed to parse response body:", e);
            }
        }
    }
}

console.log("\n\nDone!");
