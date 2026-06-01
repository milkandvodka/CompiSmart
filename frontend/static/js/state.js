export const state = {
  comparisons: [],
  activeComparisonId: "",
  activeComparison: null,
  job: null,
  jobTimer: null,
  chatBusy: false,
  messages: [],
};

export const els = {
  apiStatus: document.querySelector("#apiStatus"),
  jobForm: document.querySelector("#jobForm"),
  freshComparison: document.querySelector("#freshComparison"),
  jobError: document.querySelector("#jobError"),
  jobHint: document.querySelector("#jobHint"),
  embeddingHint: document.querySelector("#embeddingHint"),
  jobBox: document.querySelector("#jobBox"),
  runPipeline: document.querySelector("#runPipeline"),
  videoCards: document.querySelector("#videoCards"),
  chatComparison: document.querySelector("#chatComparison"),
  messages: document.querySelector("#messages"),
  chatForm: document.querySelector("#chatForm"),
  chatQuestion: document.querySelector("#chatQuestion"),
  sendChat: document.querySelector("#sendChat"),
  chatError: document.querySelector("#chatError"),
};
