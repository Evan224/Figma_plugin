// This plugin will open a window to prompt the user to enter a number, and
// it will then create that many rectangles on the screen.

// This file holds the main code for the plugins. It has access to the *document*.
// You can access browser APIs in the <script> tag inside "ui.html" which has a
// full browser environment (see documentation).

// This shows the HTML page in "ui.html".
// figma.showUI(__html__);
figma.on("drop", (event: DropEvent) => {
  console.log("[plugin] drop received!!", event);
  const { items } = event;

  const height = Number(items[3].data) || 100;
  const width = Number(items[2].data) || 100;
  const src = items[4].data;

  const rectangle = figma.createRectangle();
  rectangle.resize(height, width); // Set the size of the rectangle
  rectangle.x = event.absoluteX;
  rectangle.y = event.absoluteY;

  fetch(src)
    .then((response) => response.arrayBuffer())
    .then((buffer) => {
      const imageHash = figma.createImage(new Uint8Array(buffer)).hash;
      const imagePaint = {
        type: "IMAGE",
        scaleMode: "FILL",
        imageHash: imageHash,
      };
      rectangle.fills = [imagePaint];
    })
    .catch((error) => console.log(error));

  // if (items.length > 0 && items[0].type === 'image/svg+xml') {
  //   const data = items[0].data

  //   const newNode = figma.createNodeFromSvg(data);
  //   newNode.x = event.absoluteX;
  //   newNode.y = event.absoluteY;

  //   figma.currentPage.selection = [newNode];
  // }

  return false;
});
figma.showUI(
  `<script>window.location.href = "http://127.0.0.1:5173/"</script>"`,
);

// figma.createShapeWithText();
// figma.showUI("<h1>Hello, world!</h1>");
// Calls to "parent.postMessage" from within the HTML page will trigger this
// callback. The callback will be passed the "pluginMessage" property of the
// posted message.
// figma.closePlugin();
// figma.ui.onmessage = (message) => {
//   console.log("got this from the UI", message);
// };

// figma.on("drop", (event: DropEvent) => {
//   const { items, node, dropMetadata } = event;
//   console.log("drop", event, items, node, dropMetadata);

//   const rectangle = figma.createRectangle();
//   rectangle.resize(200, 100); // Set the size of the rectangle
//   rectangle.x = event.x;
//   rectangle.y = event.y;
//   rectangle.constrainProportions = true;
//   rectangle.constraints = { horizontal: "MIN", vertical: "MIN" };
//   // rectangle.resizable = false;
//   //   rectangle.resizable = false;

//   fetch(
//     "https://i.ytimg.com/vi/e8p1zSNmK7Q/maxresdefault.jpg",
//   )
//     .then((response) => response.arrayBuffer())
//     .then((buffer) => {
//       const imageHash = figma.createImage(new Uint8Array(buffer)).hash;
//       const imagePaint = {
//         type: "IMAGE",
//         scaleMode: "FILL",
//         imageHash: imageHash,
//       };
//       rectangle.fills = [imagePaint];
//     })
//     .catch((error) => console.log(error));

//   // create a rectangle and add the image as a fill
//   // const rect = figma.createRectangle();
//   // rect.x = dropMetadata.x;
//   // rect.y = dropMetadata.y;
//   // rect.fills = [{ type: "IMAGE", imageHash: items[0].hash, scaleMode: "FILL" }];
// });
