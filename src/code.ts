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
  // console.log("[plugin] documentChangeHandler");
  const node = figma.currentPage.findAll((node) => {
    return node.type === "RECTANGLE" && node.parent?.type === "COMPONENT" &&
      !node.name.includes("_dotted");
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

const cleanUp = () => {
  // Delete all the components and their children
  const components = figma.currentPage.findAll((node) => {
    return node.type === "COMPONENT";
  });
  components.forEach((component) => {
    // Remove children of the component
    for (const child of component.children) {
      child.remove();
    }
    // Remove the component
    component.remove();
  });
};

//todo sync problem
const createDottedComponent = (data) => {
  const { height, width, left, top, ui, type, id } = data;
  console.log("[plugin] createDottedComponent", data);
  const ui_component = figma.currentPage.findChild((node) =>
    node.name === ui && node.type === "COMPONENT"
  ) as ComponentNode;

  if (!ui_component) {
    // documentChangeHandler();
    return;
  }

  // find if already have some _dotted component, remove them
  const targetNode = figma.currentPage.findAll((node) => {
    return node.name.includes("_dotted") &&
      node.parent?.name === ui;
  });

  targetNode.forEach((node) => {
    node.remove();
  });

  // const ExistingComponent = figma.currentPage.findChild((node) => {
  //   return node.name === String(id) + "_dotted" &&
  //     node.parent?.name === ui;
  // });

  // // if exist, then remove it
  // if (ExistingComponent) {
  //   ExistingComponent.remove();
  // }

  const rectangle = figma.createRectangle();
  // ignore if ui component is not found
  if (ui_component.children.includes(rectangle)) return;
  ui_component!.appendChild(rectangle);
  rectangle.resize(width, height); // Set the size of the rectangle
  rectangle.x = left;
  rectangle.y = top;
  rectangle.name = String(id) + "_dotted";
  const blackColor: RGBA = { r: 0, g: 0, b: 0, a: 1 };

  // Create a SolidPaint object with the black color
  const blackFill: SolidPaint = {
    type: "SOLID",
    color: blackColor,
  };

  // Set the fills property of the rectangle to an array containing the SolidPaint object
  rectangle.fills = [blackFill];
};

figma.ui.onmessage = async (message) => {
  // console.log("[plugin] message", message);
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

  if (message.type === "reset") {
    // get the data ui
    const { ui, id } = message;
    // delete the target whhich parent nameis ui

    // find the target node where name = id
    // const targetNode = figma.currentPage.findChild((node) => {
    //   return node.name === id && node.parent?.name === ui;
    // }) as RectangleNode;

    // find all nodes and console its name, parent name and type
    const targetNode = figma.currentPage.findAll((node) => {
      return node.name === String(id) && node.parent?.name === ui;
    });

    // remove the target node
    targetNode.forEach((node) => {
      node.remove();
    });
    return;
  }

  if (message.type === "cleanUp") {
    cleanUp();
    return;
  }

  if (message.type === "addDottedLine") {
    console.log("[plugin] updateDottedLine", message);
    createDottedComponent(message.elementInfo);
    return;
  }

  // console.log("[plugin] message", message);
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

  // if already find a elment with the same name and parent name, then remove the element
  const targetNode = figma.currentPage.findChild((node) => {
    return node.name === String(id) + "_dotted" &&
      node.parent?.name === ui_name;
  });

  if (targetNode) {
    targetNode.remove();
  }

  await fetch(src)
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
      if (ui_component.children.includes(rectangle)) return;
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

figma.ui.resize(375, 750);
