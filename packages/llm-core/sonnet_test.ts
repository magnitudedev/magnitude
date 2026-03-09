import { b } from 'llm-core';

console.log(await b.VertexSonnetTest());

console.log("Starting stream");
const stream = b.stream.VertexSonnetTest();
console.log("Stream started")

for await (const chunk of stream) {
    console.log("Got a chunk")
    console.log("Chunk:", chunk);
}
console.log("Stream done");