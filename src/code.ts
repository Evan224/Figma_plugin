const BASIC_URL = "http://127.0.0.1:5000/";

async function invertImages(node: any) {
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
  if (items.length <= 3) {
    return false;
  }
  const height = Number(items[3].data) || 100;
  const width = Number(items[4].data) || 100;
  const pictureSrc = items[5].data;
  const name = items[7].data;
  const relativex = Number(items[6].data);
  const relativey = Number(items[2].data);
  // find if the name is already exist
  const ExistingComponent = figma.currentPage.findChild((node) => {
    return node.name === name && node.type === "COMPONENT";
  });

  if (ExistingComponent) {
    ExistingComponent.x = event.absoluteX - relativex;
    ExistingComponent.y = event.absoluteY - relativey;
    return false;
  }

  const component = figma.createComponent();
  component.resize(width, height); // Set the size of the rectangle
  component.x = event.absoluteX - relativex;
  component.y = event.absoluteY - relativey;
  component.name = name;

  fetch(pictureSrc)
    .then((response) => response.arrayBuffer())
    .then((buffer) => {
      const imageHash = figma.createImage(new Uint8Array(buffer)).hash;
      const imagePaint = {
        type: "IMAGE",
        scaleMode: "FILL",
        imageHash: imageHash,
      };
      component.fills = [imagePaint as Paint];
      figma.ui.postMessage({
        type: "uiFinished",
        data: name,
      });
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

function throttle<T extends (...args: any[]) => void>(
  func: T,
  limit: number,
): T {
  let timeout: ReturnType<typeof setTimeout> | null = null;

  return function (this: any, ...args: Parameters<T>): void {
    if (!timeout) {
      func.apply(this, args);
      timeout = setTimeout(() => {
        timeout = null;
      }, limit);
    }
  } as T;
}

function debounce<T extends (...args: any[]) => void>(
  func: T,
  delay: number,
): T {
  let timeout: ReturnType<typeof setTimeout> | null = null;

  return function (this: any, ...args: Parameters<T>): void {
    if (timeout) {
      clearTimeout(timeout);
    }

    timeout = setTimeout(() => {
      func.apply(this, args);
      timeout = null;
    }, delay);
  } as T;
}

const documentChangeHandler = debounce(() => {
  console.log("[plugin] documentChangeHandler");
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
      parent_name: node.parent?.name,
    };
  });

  figma.ui.postMessage({
    type: "uiElementChanged",
    data: element_list,
  });
}, 1000);

figma.on("documentchange", documentChangeHandler);

const getCurrentComponent = throttle(() => {
  // find the current selected component and type is component
  const currentComponent = figma.currentPage.selection.find((node) => {
    return node.type === "COMPONENT";
  });
  figma.ui.postMessage({
    type: "initializeMainPage",
    data: currentComponent?.name,
  });
}, 1000);

const setUIComponent = (UIinfo: any) => {
  // console.log(UIinfo, "UIinfo");
  const { src, ui_name, height, width } = UIinfo;

  const ExistingComponent = figma.currentPage.findChild((node) => {
    return node.name === ui_name && node.type === "COMPONENT";
  });

  if (ExistingComponent) {
    return false;
  }

  fetch(src)
    .then((response) => response.arrayBuffer())
    .then((buffer) => {
      const component = figma.createComponent();
      component.resize(width, height); // Set the size of the rectangle
      component.x = figma.viewport.center.x - width / 2;
      component.y = figma.viewport.center.y - height / 2;
      component.name = ui_name;
      const imageHash = figma.createImage(new Uint8Array(buffer)).hash;
      const imagePaint = {
        type: "IMAGE",
        scaleMode: "FILL",
        imageHash: imageHash,
      };
      component.fills = [imagePaint as Paint];
      figma.ui.postMessage({
        type: "uiFinished",
        data: ui_name,
      });
    })
    .catch((error) => console.log(error));

  return false;
};

figma.ui.onmessage = (message) => {
  if (message.type === "initializeList") {
    documentChangeHandler();
    return;
  }

  if (message.type === "initializeMainPage") {
    getCurrentComponent();
    return;
  }

  if (message.type === "setUIComponent") {
    setUIComponent(message?.uiInfo);
    return;
  }

  console.log("[plugin] message", message);
  const elementInfo = message.elementInfo;
  const { element_name, height, width, left, top, ui_name, src, id } =
    elementInfo;
  const ui_component = figma.currentPage.findChild((node) =>
    node.name === ui_name && node.type === "COMPONENT"
  ) as ComponentNode;

  if (!ui_component) {
    documentChangeHandler();
    return;
  }

  fetch(src)
    .then((response) => response.arrayBuffer())
    .then((buffer) => {
      const imageHash = figma.createImage(new Uint8Array(buffer)).hash;
      const imagePaint = {
        type: "IMAGE",
        scaleMode: "FILL",
        imageHash: imageHash,
      };
      const rectangle = figma.createRectangle();
      // ignore if ui component is not found
      ui_component!.appendChild(rectangle);
      rectangle.fills = [imagePaint as Paint];
      rectangle.resize(width, height); // Set the size of the rectangle
      rectangle.x = left;
      rectangle.y = top;
      rectangle.name = String(id);
      if (message.type === "initialSetUp") {
        // rectangle.parent = figma.currentPage.selection[0];
      } else {
        figma.currentPage.selection = [rectangle];
      }
    })
    .catch((error) => console.log(error));
};

//# resize the UI to 375 675
figma.showUI(
  `<script>window.location.href = "http://127.0.0.1:5173/"</script>"`,
);

figma.ui.resize(375, 500);
