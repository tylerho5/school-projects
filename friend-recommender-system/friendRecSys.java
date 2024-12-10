// Tyler Ho
// CS-439
// Homework 1
// Oct 16, 2024

import java.io.IOException;
import java.util.*;
import java.util.HashMap;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;.
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class friendRecSys extends Configured implements Tool {

    public static void main(String[] args) throws Exception {
        System.out.println(Arrays.toString(args));
        int res = ToolRunner.run(new Configuration(), new question_1(), args);
        System.exit(res);
    }

    @Override
    public int run(String[] args) throws Exception {
        System.out.println(Arrays.toString(args));

        Job job = new Job(getConf(), "HowManyMutuals");
        job.setJarByClass(question_1.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        job.setMapperClass(Map.class);
        job.setReducerClass(Reduce.class);

        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        job.waitForCompletion(true);
        return 0;
    }

    public static class Map extends Mapper<LongWritable, Text, Text, Text> {
        private Text userID = new Text();
        private Text friendID = new Text();

        @Override
        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            // Each line: <User><TAB><Friends>
            String[] line = value.toString().split("\\t");
            if (line.length != 2) {
                return;
            }
            String user = line[0];
            String[] friends = line[1].split(",");

            // Emit existing friends of user
            for (int i = 0; i < friends.length; i++) {
                // Emit user-friend pair
                userID.set(user);
                friendID.set("friend," + friends[i]);
                context.write(userID, friendID);

                // accounts for friend-friend relationship
                // the friendship status goes both ways
                userID.set(friends[i]);
                friendID.set("friend," + user);
                context.write(userID, friendID);
            }

            // Emit potential friends via mutual friends
            for (int i = 0; i < friends.length; i++) {
                for (int j = i + 1; j < friends.length; j++) {
                    // Friend of user
                    userID.set(friends[i]);
                    friendID.set("potentialFriend," + friends[j]);
                    context.write(userID, friendID);

                    // friend of friend
                    userID.set(friends[j]);
                    friendID.set("potentialFriend," + friends[i]);
                    context.write(userID, friendID);
                }
            }
        }
    }

    public static class Reduce extends Reducer<Text, Text, Text, Text> {
        // stops recommendations at count 10
        private static final int MAX_RECS = 10;

        @Override
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            
            Set<String> friends = new HashSet<>();
            // hashes future mutuals to hashmap
            HashMap<String, Integer> mutualsToAdd = new HashMap<>();

            for (Text val : values) {
                String[] tokens = val.toString().split(",");
                String potentialFriend = tokens[1];

                // checks if user is friend
                if (tokens[0].equals("friend")) {
                    friends.add(tokens[1]);

                // or potential friend
                } else if (tokens[0].equals("potentialFriend")) {

                    // adds potential friend to hashmap
                    if (!friends.contains(potentialFriend) && !potentialFriend.equals(key.toString())) {

                        // counts number of potential friends
                        Integer count = mutualsToAdd.get(potentialFriend);
                        if (count == null) {
                            mutualsToAdd.put(potentialFriend, 1);
                        } else {
                            mutualsToAdd.put(potentialFriend, count + 1);
                        }
                    }
                }
            }

            // Sort recommendations
            List<Entry<String, Integer>> recList = new ArrayList<>(mutualsToAdd.entrySet());
            Collections.sort(recList, new Comparator<Entry<String, Integer>>() {
                public int compare(Entry<String, Integer> user1, Entry<String, Integer> user2) {

                    // arranges id's by number of mutual friends
                    int compar = user2.getValue().compareTo(user1.getValue());
                    if (compar != 0) {
                        return compar;
                    }

                    Integer userId1 = Integer.parseInt(user1.getKey());
                    Integer userId2 = Integer.parseInt(user2.getKey());
                    return userId1.compareTo(userId2);
                    }
                }
            );

            // Build the recommendation string
            StringBuilder sb = new StringBuilder();
            int i = 0;
            for (Entry<String, Integer> entry : recList) {
                if (i >= MAX_RECS) {
                    break;
                }
                if (i > 0) {
                    sb.append(",");
                }
                sb.append(entry.getKey());
                i++;
            }
            context.write(key, new Text(sb.toString()));
        }
    }
}