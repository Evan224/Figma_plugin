const BASIC_URL = "http://127.0.0.1:5000/";

async function invertImages(node) {
  // console.log("[plugin] invertImages", node);
  for (const paint of node.fills) {
    if (paint.type === "IMAGE") {
      // Get the (encoded) bytes for this image.
      const image = figma.getImageByHash(paint.imageHash);
      const bytes = await image?.getBytesAsync() ?? new Uint8Array(0);
      return bytes;
    }
  }
  return null;
}

figma.on("drop", (event: DropEvent) => {
  const { items } = event;
  console.log("[plugin] items", items);
  const height = Number(items[3].data) || 100;
  const width = Number(items[2].data) || 100;
  const pictureSrc = items[4].data;

  const rectangle = figma.createRectangle();
  rectangle.resize(height, width); // Set the size of the rectangle
  rectangle.x = event.absoluteX;
  rectangle.y = event.absoluteY;
  console.log("[plugin] rectangle", pictureSrc);
  fetch(pictureSrc)
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

  return false;
});

figma.on("selectionchange", async () => {
  // Check if the current selection is a rectangle
  if (
    figma.currentPage.selection.length === 1 &&
    figma.currentPage.selection[0].type === "RECTANGLE"
  ) {
    figma.ui.postMessage({
      type: "uiSelectionChanged",
      data: await invertImages(figma.currentPage.selection[0]),
    });
  }
});

figma.showUI(
  `<script>window.location.href = "http://127.0.0.1:5173/"</script>"`,
);
