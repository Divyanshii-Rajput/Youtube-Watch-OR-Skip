import React, { useState } from "react";
import { FaYoutube, FaCheckCircle, FaTimesCircle } from "react-icons/fa";

function App() {
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleAnalyze = async () => {
    if (!url) {
      alert("Please enter a YouTube URL");
      return;
    }
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch("http://127.0.0.1:8000/predict/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      const data = await res.json();
      setResult(data);
    } catch (error) {
      alert("Error fetching prediction");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-pink-100 to-pink-200 p-6">
      <div className="bg-white shadow-lg rounded-2xl p-8 w-full max-w-xl">
        <div className="flex items-center justify-center gap-3 mb-4">
          <FaYoutube className="text-red-500 text-3xl" />
          <h1 className="text-2xl font-bold text-gray-700">
            YouTube Watch-or-Skip Predictor 🎥
          </h1>
        </div>

        <div className="flex gap-2 mb-6">
          <input
            type="text"
            placeholder="Enter YouTube video link..."
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            className="flex-1 p-3 border border-pink-300 rounded-xl focus:ring-2 focus:ring-pink-400 outline-none"
          />
          <button
            onClick={handleAnalyze}
            disabled={loading}
            className="bg-pink-500 text-white px-4 py-3 rounded-xl hover:bg-pink-600 transition-all"
          >
            {loading ? "Analyzing..." : "Analyze"}
          </button>
        </div>

        {result && (
          <div
            className={`mt-6 p-6 rounded-2xl text-center shadow-md ${
              result.decision.toLowerCase().includes("watch")
                ? "bg-green-100 border-2 border-green-400"
                : "bg-red-100 border-2 border-red-400"
            }`}
          >
            <div className="flex items-center justify-center gap-2 text-2xl font-semibold">
              {result.decision.toLowerCase().includes("watch") ? (
                <FaCheckCircle className="text-green-500" />
              ) : (
                <FaTimesCircle className="text-red-500" />
              )}
              <span>{result.decision}</span>
            </div>
            <p className="text-gray-600 mt-2 text-sm">
              Sentiment Score: {result.features?.sentiment?.toFixed(2)} | Likes:{" "}
              {result.features?.likes} | Comments:{" "}
              {result.features?.comments}
            </p>
          </div>
        )}
      </div>
      <p className="text-gray-500 mt-6 text-sm">
       
      </p>
    </div>
  );
}

export default App;
