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
  const name = items[5].data;

  const component = figma.createComponent();
  component.resize(height, width); // Set the size of the rectangle
  component.x = event.absoluteX;
  component.y = event.absoluteY;
  component.name = name;
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
      component.fills = [imagePaint];
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

figma.on("documentchange", () => {
  const node = figma.currentPage.findAll((node) => {
    return node.type === "RECTANGLE";
  });
  const element_list = node.map((node) => {
    return {
      height: node.height,
      width: node.width,
      left: node.x,
      top: node.y,
      id: node.name,
    };
  });

  figma.ui.postMessage({
    type: "uiElementChanged",
    data: element_list,
  });
});

figma.ui.onmessage = (message) => {
  const elementInfo = message.elementInfo;
  const { element_name, height, width, left, top, ui_name, src, id } =
    elementInfo;
  const ui_component = figma.currentPage.findChild((node) =>
    node.name === ui_name && node.type === "COMPONENT"
  );

  const rectangle = figma.createRectangle();
  rectangle.resize(height, width); // Set the size of the rectangle
  rectangle.x = left;
  rectangle.y = top;
  rectangle.name = String(id);

  ui_component!.appendChild(rectangle);
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
};

//# resize the UI to 375 675
figma.showUI(
  `<script>window.location.href = "http://127.0.0.1:5173/"</script>"`,
);

figma.ui.resize(375, 675);
