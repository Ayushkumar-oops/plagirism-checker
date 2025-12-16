package plaragismchecker;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.*;
import java.text.DecimalFormat;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;


public class PlagiarismChecker {

    static class Document {
        String name;
        Map<String, Integer> termFreq = new HashMap<>();
        int totalTerms = 0;

        Document(String name) {
            this.name = name;
        }

        void tokenize(String text, Set<String> stopwords) {
            termFreq.clear();
            totalTerms = 0;
            // Tokenization: extract letter sequences (unicode-aware), allow internal apostrophes
            Pattern pattern = Pattern.compile("\\p{L}+(?:'\\p{L}+)?");
            Matcher m = pattern.matcher(text.toLowerCase(Locale.ROOT));
            while (m.find()) {
                String token = m.group();
                if (stopwords != null && stopwords.contains(token)) continue;
                termFreq.put(token, termFreq.getOrDefault(token, 0) + 1);
                totalTerms++;
            }
        }


        double tf(String token) {
            if (totalTerms == 0) return 0.0;
            return termFreq.getOrDefault(token, 0) / (double) totalTerms;
        }
    }

    // Compute cosine similarity between two double vectors
    static double cosineSimilarity(double[] a, double[] b) {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na == 0 || nb == 0) return 0.0;
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }

    // Load all .txt files from directory (recursive)
    static List<Path> listTxtFiles(Path dir) throws IOException {
        try (var stream = Files.walk(dir)) {
            return stream.filter(Files::isRegularFile)
                    .filter(p -> p.toString().toLowerCase().endsWith(".txt"))
                    .collect(Collectors.toList());
        }
    }

    // Build IDF map: idf(token) = log((N) / (1 + docFreq))
    static Map<String, Double> computeIdf(List<Document> documents, Set<String> vocab) {
        Map<String, Integer> docFreq = new HashMap<>();
        for (String term : vocab) {
            int count = 0;
            for (Document d : documents) if (d.termFreq.containsKey(term)) count++;
            docFreq.put(term, count);
        }
        Map<String, Double> idf = new HashMap<>();
        int N = documents.size();
        for (String term : vocab) {
            int df = docFreq.getOrDefault(term, 0);
            // smoothing with +1 to avoid divide-by-zero and dampen very rare terms
            idf.put(term, Math.log((double) N / (1 + df)) + 1.0);
        }
        return idf;
    }

    // Build TF-IDF vector for a document given ordered vocabulary
    static double[] buildTfIdfVector(Document d, List<String> vocabOrder, Map<String, Double> idf) {
        double[] vec = new double[vocabOrder.size()];
        for (int i = 0; i < vocabOrder.size(); i++) {
            String term = vocabOrder.get(i);
            double tf = d.tf(term);           // normalized term frequency
            double w = tf * idf.getOrDefault(term, 0.0);
            vec[i] = w;
        }
        return vec;
    }

    // Some common English stopwords (small set — expandable)
    static Set<String> defaultStopwords() {
        String[] s = new String[] {
                "the","is","in","and","a","an","to","of","that","this","it","for","on","with","as","by","are","from","at","be","or","was","were","which"
        };
        return new HashSet<>(Arrays.asList(s));
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        List<Document> documents = new ArrayList<>();
        List<Path> loadedPaths = new ArrayList<>();
        boolean useStopwords = true;
        double threshold = 0.7; // default threshold for flags

        while (true) {
            System.out.println("\nOptions:");
            System.out.println("1 - Add directory of .txt files (recursive)");
            System.out.println("2 - Toggle stopwords (currently " + (useStopwords ? "ON" : "OFF") + ")");
            System.out.println("3 - Compute similarity");
            System.out.println("4 - Set plagiarism threshold (current \" + threshold + \")");
            System.out.println("5 - Export report (CSV + flags)");
            System.out.println("6 - List loaded files");
            System.out.println("7 - Clear loaded files");
            System.out.println("8 - Exit");
            System.out.print("Choose an option: ");

            String line = scanner.nextLine().trim();
            if (line.isEmpty()) continue;
            int option;
            try { option = Integer.parseInt(line); } catch (NumberFormatException e) { System.out.println("Invalid option."); continue; }

            if (option == 1) {
                System.out.print("Enter directory path: ");
                String dir = scanner.nextLine().trim();
                Path dirPath = Paths.get(dir);
                if (!Files.exists(dirPath) || !Files.isDirectory(dirPath)) {
                    System.out.println("Directory not found: " + dir);
                    continue;
                }
                try {
                    List<Path> files = listTxtFiles(dirPath);
                    if (files.isEmpty()) { System.out.println("No .txt files found under directory."); continue; }
                    documents.clear();
                    loadedPaths.clear();
                    Set<String> stopwords = useStopwords ? defaultStopwords() : null;
                    for (Path p : files) {
                        try {
                            String content = Files.readString(p);
                            Document d = new Document(p.getFileName().toString());
                            d.tokenize(content, stopwords);
                            documents.add(d);
                            loadedPaths.add(p);
                        } catch (IOException ioe) {
                            System.out.println("Failed to read file: " + p + " (" + ioe.getMessage() + ")");
                        }
                    }
                    System.out.println("Loaded " + documents.size() + " documents.");
                } catch (IOException e) {
                    System.out.println("Error listing files: " + e.getMessage());
                }
            } else if (option == 2) {
                useStopwords = !useStopwords;
                System.out.println("Stopwords removal is now " + (useStopwords ? "ON" : "OFF"));
            } else if (option == 3) {
                if (documents.size() < 2) {
                    System.out.println("Add at least two documents first.");
                    continue;
                }
                // Build vocabulary in deterministic order (LinkedHashSet -> List)
                LinkedHashSet<String> vocab = new LinkedHashSet<>();
                for (Document d : documents) vocab.addAll(d.termFreq.keySet());
                List<String> vocabOrder = new ArrayList<>(vocab);
                Map<String, Double> idf = computeIdf(documents, vocab);
                double[][] sim = new double[documents.size()][documents.size()];
                for (int i = 0; i < documents.size(); i++) {
                    double[] vi = buildTfIdfVector(documents.get(i), vocabOrder, idf);
                    for (int j = i; j < documents.size(); j++) {
                        double[] vj = buildTfIdfVector(documents.get(j), vocabOrder, idf);
                        double s = cosineSimilarity(vi, vj);
                        sim[i][j] = s;
                        sim[j][i] = s;
                    }
                }
                // Display matrix
                DecimalFormat df = new DecimalFormat("#0.000");
                System.out.println("\nSimilarity Matrix (TF-IDF, cosine):");
                System.out.print("Doc\\Doc");
                for (Document d : documents) System.out.print("\t" + d.name);
                System.out.println();
                for (int i = 0; i < documents.size(); i++) {
                    System.out.print(documents.get(i).name);
                    for (int j = 0; j < documents.size(); j++) {
                        System.out.print("\t" + df.format(sim[i][j]));
                    }
                    System.out.println();
                }
            } else if (option == 4) {
                System.out.print("Enter threshold between 0 and 1 (current " + threshold + "): ");
                String v = scanner.nextLine().trim();
                try {
                    double t = Double.parseDouble(v);
                    if (t < 0 || t > 1) throw new NumberFormatException();
                    threshold = t;
                    System.out.println("Threshold set to " + threshold);
                } catch (NumberFormatException e) {
                    System.out.println("Invalid threshold. Must be a number between 0 and 1.");
                }
            } else if (option == 5) {
                if (documents.size() < 2) {
                    System.out.println("Add documents and compute similarity first.");
                    continue;
                }
                System.out.print("Enter CSV output filename (default similarity_matrix.csv): ");
                String csvName = scanner.nextLine().trim();
                if (csvName.isEmpty()) csvName = "similarity_matrix.csv";
                String flagsName = "plagiarism_flags.txt";
                // Recompute to ensure latest
                LinkedHashSet<String> vocab = new LinkedHashSet<>();
                for (Document d : documents) vocab.addAll(d.termFreq.keySet());
                List<String> vocabOrder = new ArrayList<>(vocab);
                Map<String, Double> idf = computeIdf(documents, vocab);
                double[][] sim = new double[documents.size()][documents.size()];
                for (int i = 0; i < documents.size(); i++) {
                    double[] vi = buildTfIdfVector(documents.get(i), vocabOrder, idf);
                    for (int j = i; j < documents.size(); j++) {
                        double[] vj = buildTfIdfVector(documents.get(j), vocabOrder, idf);
                        double s = cosineSimilarity(vi, vj);
                        sim[i][j] = s;
                        sim[j][i] = s;
                    }
                }
                try (PrintWriter csv = new PrintWriter(csvName);
                     PrintWriter flags = new PrintWriter(flagsName)) {
                    // CSV header
                    csv.print("\"Document\"");
                    for (Document d : documents) csv.print("," + quoteCsv(d.name));
                    csv.println();
                    DecimalFormat df = new DecimalFormat("#0.0000");
                    for (int i = 0; i < documents.size(); i++) {
                        csv.print(quoteCsv(documents.get(i).name));
                        for (int j = 0; j < documents.size(); j++) {
                            csv.print("," + df.format(sim[i][j]));
                        }
                        csv.println();
                    }
                    // Flags
                    for (int i = 0; i < documents.size(); i++) {
                        for (int j = i + 1; j < documents.size(); j++) {
                            if (sim[i][j] >= threshold) {
                                flags.printf("Potential plagiarism: %s and %s (%.4f)%n",
                                        documents.get(i).name, documents.get(j).name, sim[i][j]);
                            }
                        }
                    }
                    System.out.println("Exported: " + csvName + " and " + flagsName);
                } catch (IOException e) {
                    System.out.println("Error writing reports: " + e.getMessage());
                }

            } else if (option == 6) {
                if (loadedPaths.isEmpty()) System.out.println("No files loaded.");
                else {
                    System.out.println("Loaded files:");
                    for (Path p : loadedPaths) System.out.println(" - " + p.toString());
                }
            } else if (option == 7) {
                documents.clear();
                loadedPaths.clear();
                System.out.println("Cleared loaded files.");
            } else if (option == 8) {
                System.out.println("Exiting.");
                break;
            } else {
                System.out.println("Invalid option.");
            }
        }

        scanner.close();
    }

    static String quoteCsv(String s) {
        return "\"" + s.replace("\"", "\"\"") + "\"";
    }
}

